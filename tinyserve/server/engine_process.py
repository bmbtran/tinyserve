"""Run the engine core in its own process (FIXPLAN D2).

Why: with the engine on a thread, the HTTP side (asyncio, JSON, SSE, detokenizing)
and `engine.step()` share one Python GIL, which cost ~21% of throughput in the
profile. vLLM runs its engine core in a separate process for the same reason.

    HTTP process                                   engine process (spawned)
    handlers --("add"/"abort", via in_q)-------->  drain in_q, engine.step()
    reader thread <--("step", outputs, metrics)--  one message per step for ALL sequences

The engine process blocks on in_q when it has no work (no busy spinning).
Same interface as AsyncEngine, so api.py works with either.
"""

from __future__ import annotations

import asyncio
import itertools
import multiprocessing as mp
import queue
import threading
import traceback
from dataclasses import dataclass
from typing import AsyncIterator

from tinyserve.config import EngineConfig
from tinyserve.sampling import SamplingParams


def _engine_main(cfg: EngineConfig, in_q, out_q) -> None:
    """Entry point of the engine process."""
    try:
        from tinyserve.engine import LLMEngine

        eng = LLMEngine(cfg, load_tokenizer=False)
    except BaseException as e:  # noqa: BLE001
        out_q.put(("fatal", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
        return
    out_q.put(("ready", {"num_kv_blocks": eng.block_manager.num_blocks, "metrics": eng.metrics()}))
    live: dict[int, tuple[int, object]] = {}  # rid -> (seq_id, seq)
    by_seq: dict[int, int] = {}  # seq_id -> rid
    while True:
        msgs = []
        if not eng.has_work():
            out_q.put(("metrics", eng.metrics()))
            msgs.append(in_q.get())  # idle: block until a request arrives
        while True:
            try:
                msgs.append(in_q.get_nowait())
            except queue.Empty:
                break
        for m in msgs:
            if m[0] == "stop":
                return
            if m[0] == "add":
                _, rid, ids, params = m
                try:
                    seq = eng.add_request(ids, params)
                    live[rid] = (seq.seq_id, seq)
                    by_seq[seq.seq_id] = rid
                except Exception as e:  # noqa: BLE001
                    out_q.put(("error", rid, str(e)))
            elif m[0] == "abort":
                entry = live.pop(m[1], None)
                if entry is not None:
                    by_seq.pop(entry[0], None)
                    eng.abort(entry[1])
        if not eng.has_work():
            continue
        try:
            batch, _ = eng.step()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            for rid, (_, seq) in list(live.items()):
                eng.abort(seq)
                out_q.put(("error", rid, f"engine step failed: {e}"))
            live.clear()
            by_seq.clear()
            continue
        if batch is None:
            continue
        outs = []
        for seq in batch.seqs:
            rid = by_seq.get(seq.seq_id)
            if rid is None:
                continue
            outs.append((rid, seq.completion_token_ids[getattr(seq, "_emitted", 0):], seq.is_finished, seq.finish_reason,
                         seq.num_prompt_tokens, seq.num_completion_tokens))
            seq._emitted = seq.num_completion_tokens
            if seq.is_finished:
                live.pop(rid, None)
                by_seq.pop(seq.seq_id, None)
        out_q.put(("step", outs, eng.metrics()))


@dataclass
class _Request:
    queue: asyncio.Queue
    loop: asyncio.AbstractEventLoop
    done: bool = False

    def push(self, item) -> None:
        self.loop.call_soon_threadsafe(self.queue.put_nowait, item)


class ProcessAsyncEngine:
    def __init__(self, cfg: EngineConfig, tokenizer=None, startup_timeout_s: float = 1800):
        self.cfg = cfg.validate()
        self.tokenizer = tokenizer
        ctx = mp.get_context("spawn")
        self._in, self._out = ctx.Queue(), ctx.Queue()
        self._proc = ctx.Process(target=_engine_main, args=(cfg, self._in, self._out), name="tinyserve-engine-core", daemon=True)
        self._proc.start()
        msg = self._out.get(timeout=startup_timeout_s)
        if msg[0] != "ready":
            raise RuntimeError(f"engine process failed to start:\n{msg[1]}")
        self.num_kv_blocks = msg[1]["num_kv_blocks"]
        self._metrics = msg[1]["metrics"]
        self._reqs: dict[int, _Request] = {}
        self._lock = threading.Lock()
        self._ids = itertools.count()
        self._reader = threading.Thread(target=self._read_loop, name="tinyserve-engine-reader", daemon=True)
        self._reader.start()

    def metrics(self) -> dict:
        return self._metrics

    def shutdown(self) -> None:
        self._in.put(("stop",))
        self._proc.join(timeout=10)
        if self._proc.is_alive():
            self._proc.terminate()

    def _read_loop(self) -> None:
        while True:
            try:
                msg = self._out.get()
            except (EOFError, OSError):
                return
            kind = msg[0]
            if kind == "step":
                self._metrics = msg[2]  # update before pushing tokens, so /metrics is never behind
                for rid, toks, finished, reason, n_prompt, n_comp in msg[1]:
                    with self._lock:
                        req = self._reqs.pop(rid, None) if finished else self._reqs.get(rid)
                    if req is not None:
                        req.push({"token_ids": toks, "finished": finished, "finish_reason": reason if finished else None,
                                  "num_prompt_tokens": n_prompt, "num_completion_tokens": n_comp})
            elif kind == "metrics":
                self._metrics = msg[1]
            elif kind == "error":
                with self._lock:
                    req = self._reqs.pop(msg[1], None)
                if req is not None:
                    req.push(ValueError(msg[2]))

    async def generate(self, prompt_ids: list[int], params: SamplingParams) -> AsyncIterator[dict]:
        rid = next(self._ids)
        req = _Request(asyncio.Queue(), asyncio.get_running_loop())
        with self._lock:
            self._reqs[rid] = req
        self._in.put(("add", rid, list(prompt_ids), params))
        try:
            while True:
                item = await req.queue.get()
                if isinstance(item, BaseException):
                    req.done = True
                    raise item
                yield item
                if item["finished"]:
                    req.done = True
                    return
        finally:
            if not req.done:  # client went away: free the sequence's KV
                with self._lock:
                    self._reqs.pop(rid, None)
                self._in.put(("abort", rid))
