"""Bridge between asyncio request handlers and the (synchronous) engine.

One background thread owns the LLMEngine and runs `engine.step()` while
there is work. Handlers never touch the engine directly:

    handler  --(pending list + Event)-->  engine thread
    handler  <--(asyncio.Queue via loop.call_soon_threadsafe)--  engine thread

After every step, each sequence in the batch gets ONE output event carrying
the tokens it gained in that step (1 for decode, up to 16 for spec decode).
When idle, the thread blocks on a threading.Event (no busy spinning).
"""

from __future__ import annotations

import asyncio
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from tinyserve.engine import LLMEngine
from tinyserve.sampling import SamplingParams


@dataclass
class _Request:
    prompt_ids: list[int]
    params: SamplingParams
    queue: asyncio.Queue
    loop: asyncio.AbstractEventLoop
    seq: Any = None
    emitted: int = 0
    done: bool = False

    def push(self, item) -> None:
        self.loop.call_soon_threadsafe(self.queue.put_nowait, item)


@dataclass
class _Shared:
    pending: list = field(default_factory=list)
    aborts: list = field(default_factory=list)


class AsyncEngine:
    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._shared = _Shared()
        self._live: dict[int, _Request] = {}
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name="tinyserve-engine", daemon=True)
        self._thread.start()

    @property
    def tokenizer(self):
        return self.engine.tokenizer

    def metrics(self) -> dict:
        return self.engine.metrics()

    def shutdown(self) -> None:
        self._stop = True
        self._wake.set()
        self._thread.join(timeout=5)

    async def generate(self, prompt_ids: list[int], params: SamplingParams) -> AsyncIterator[dict]:
        """Yields {"token_ids": [...new...], "finished": bool, "finish_reason", "num_prompt_tokens"}."""
        req = _Request(list(prompt_ids), params, asyncio.Queue(), asyncio.get_running_loop())
        with self._lock:
            self._shared.pending.append(req)
        self._wake.set()
        try:
            while True:
                item = await req.queue.get()
                if isinstance(item, BaseException):
                    raise item
                yield item
                if item["finished"]:
                    req.done = True
                    return
        finally:
            if not req.done:  # client went away: free the sequence's KV
                with self._lock:
                    self._shared.aborts.append(req)
                self._wake.set()

    # ------------------------------------------------------------ engine thread
    def _loop(self) -> None:
        eng = self.engine
        while not self._stop:
            self._wake.clear()
            with self._lock:
                pending, self._shared.pending = self._shared.pending, []
                aborts, self._shared.aborts = self._shared.aborts, []
            for req in pending:
                try:
                    req.seq = eng.add_request(req.prompt_ids, req.params)
                    self._live[req.seq.seq_id] = req
                except Exception as e:  # noqa: BLE001  (e.g. prompt too long)
                    req.push(ValueError(str(e)))
            for req in aborts:
                if req.seq is not None and self._live.pop(req.seq.seq_id, None) is not None:
                    eng.abort(req.seq)
            if not eng.has_work():
                self._wake.wait()
                continue
            try:
                batch, _ = eng.step()
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                for req in list(self._live.values()):
                    eng.abort(req.seq)
                    req.push(RuntimeError(f"engine step failed: {e}"))
                self._live.clear()
                continue
            if batch is None:
                continue
            for seq in batch.seqs:
                req = self._live.get(seq.seq_id)
                if req is None:
                    continue
                toks = seq.completion_token_ids
                new = toks[req.emitted :]
                req.emitted = len(toks)
                req.push({
                    "token_ids": new,
                    "finished": seq.is_finished,
                    "finish_reason": seq.finish_reason if seq.is_finished else None,
                    "num_prompt_tokens": seq.num_prompt_tokens,
                    "num_completion_tokens": len(toks),
                })
                if seq.is_finished:
                    del self._live[seq.seq_id]
