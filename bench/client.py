"""Async streaming load generator for any OpenAI-compatible server
(tinyserve and vLLM get byte-identical request payloads).

Closed-loop concurrency: at most `concurrency` requests are in flight; a new
one starts as soon as one finishes (like `vllm bench serve --max-concurrency`).

Per request we record: send time, first-token time (first chunk with
non-empty text), every chunk time, output token count (from `usage`), and errors.
Metric definitions mirror vllm/benchmarks/serve.py:
  TTFT = first chunk - send          ITL = gaps between consecutive chunks
  TPOT = (e2e - TTFT) / (output_tokens - 1)
With speculative decoding one chunk can carry several tokens, so ITL is a
per-chunk gap while TPOT is per token.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time

import httpx


def build_payload(req: dict, model: str, endpoint: str) -> dict:
    base = {"model": model, "max_tokens": req["max_tokens"], "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    if req.get("ignore_eos"):
        base["ignore_eos"] = True
    if endpoint == "chat":
        return {**base, "messages": req["messages"], "chat_template_kwargs": {"enable_thinking": False}}
    return {**base, "prompt": req["prompt"]}


async def _one(client: httpx.AsyncClient, url: str, payload: dict, keep_text: bool) -> dict:
    rec = {"send": time.perf_counter(), "first": None, "chunks": [], "output_tokens": None, "error": None, "text": ""}
    try:
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                rec["error"] = f"HTTP {resp.status_code}: {(await resp.aread())[:300]!r}"
                return rec
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                obj = json.loads(data)
                if "error" in obj:
                    rec["error"] = str(obj["error"])
                    break
                if obj.get("usage"):
                    rec["output_tokens"] = obj["usage"]["completion_tokens"]
                for ch in obj.get("choices") or []:
                    text = ch.get("text")
                    if text is None:
                        text = (ch.get("delta") or {}).get("content")
                    if text:
                        now = time.perf_counter()
                        if rec["first"] is None:
                            rec["first"] = now
                        rec["chunks"].append(now)
                        if keep_text:
                            rec["text"] += text
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["end"] = time.perf_counter()
    if rec["output_tokens"] is None and rec["error"] is None:
        rec["error"] = "no usage in stream"
    if not keep_text:
        rec.pop("text")
    return rec


async def run_load(base_url: str, requests: list[dict], concurrency: int, model: str, endpoint: str = "completions",
                   keep_text: bool = False, timeout_s: float = 1800, transport=None) -> dict:
    url = base_url.rstrip("/") + ("/v1/chat/completions" if endpoint == "chat" else "/v1/completions")
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency + 4)
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s), limits=limits, transport=transport) as client:
        async def guarded(r):
            async with sem:
                return await _one(client, url, build_payload(r, model, endpoint), keep_text)

        t0 = time.perf_counter()
        recs = await asyncio.gather(*(guarded(r) for r in requests))
        wall = time.perf_counter() - t0
    return {"records": recs, "wall_s": wall, "concurrency": concurrency}


def _pct(xs: list[float]) -> dict:
    if not xs:
        return {"p50": None, "p90": None, "p99": None, "mean": None}
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]  # noqa: E731
    return {"p50": round(q(50), 3), "p90": round(q(90), 3), "p99": round(q(99), 3), "mean": round(statistics.fmean(xs), 3)}


def summarize(run: dict) -> dict:
    recs = run["records"]
    ok = [r for r in recs if r["error"] is None and r["first"] is not None]
    ttft = [(r["first"] - r["send"]) * 1000 for r in ok]
    itl = [(b - a) * 1000 for r in ok for a, b in zip(r["chunks"], r["chunks"][1:])]
    tpot = [((r["end"] - r["first"]) * 1000) / (r["output_tokens"] - 1) for r in ok if r["output_tokens"] and r["output_tokens"] > 1]
    out_tok = sum(r["output_tokens"] or 0 for r in ok)
    chunks = sum(len(r["chunks"]) for r in ok)
    return {
        "concurrency": run["concurrency"],
        "num_requests": len(recs),
        "errors": len(recs) - len(ok),
        "error_samples": [r["error"] for r in recs if r["error"]][:3],
        "ttft_ms": _pct(ttft),
        "itl_ms": _pct(itl),
        "tpot_ms": _pct(tpot),
        "e2e_ms": _pct([(r["end"] - r["send"]) * 1000 for r in ok]),
        "output_tokens": out_tok,
        "tokens_per_chunk": round(out_tok / chunks, 3) if chunks else None,
        "output_tok_s": round(out_tok / run["wall_s"], 2),
        "req_s": round(len(ok) / run["wall_s"], 3),
        "wall_s": round(run["wall_s"], 2),
    }


def wait_for_health(base_url: str, timeout_s: float = 900, proc=None) -> float:
    """Poll /health until 200; returns seconds waited. Raises if the server process died."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode} before becoming healthy")
        try:
            if httpx.get(base_url.rstrip("/") + "/health", timeout=2).status_code == 200:
                return time.time() - t0
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise TimeoutError("server did not become healthy")
