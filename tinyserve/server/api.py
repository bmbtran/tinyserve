"""OpenAI-compatible HTTP server (FastAPI + server-sent events).

    python -m tinyserve.server.api --model Qwen/Qwen3-0.6B --port 8000

Endpoints: POST /v1/completions, POST /v1/chat/completions, GET /v1/models,
GET /health, GET /metrics (JSON: running, waiting, kv_usage, prefix_hit_rate, spec_tau, ...).
"""

from __future__ import annotations

import argparse
import json
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from tinyserve.sampling import SamplingParams
from tinyserve.server.async_engine import AsyncEngine
from tinyserve.server.protocol import ChatCompletionRequest, CompletionRequest


class IncrementalDetokenizer:
    """Emit only the newly decoded text, decoding a short window of recent
    tokens instead of the whole output every time (vLLM's approach; decoding
    everything each chunk is O(n^2) and holds the GIL the engine thread needs).

    prefix_offset..read_offset is text already emitted; we decode the window
    twice (with and without the new tokens) and emit the difference. A
    trailing U+FFFD means the last token ended mid UTF-8 character, so it is
    held back until the next token completes it (or the request ends)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids: list[int] = []
        self.prefix_offset = 0
        self.read_offset = 0

    def _decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def add(self, new_ids: list[int], final: bool) -> str:
        self.ids.extend(new_ids)
        prefix_text = self._decode(self.ids[self.prefix_offset : self.read_offset])
        new_text = self._decode(self.ids[self.prefix_offset :])
        if len(new_text) <= len(prefix_text) or (new_text.endswith("�") and not final):
            return ""
        delta = new_text[len(prefix_text) :]
        # The tokens of this chunk become the left context of the next one.
        self.prefix_offset, self.read_offset = self.read_offset, len(self.ids)
        return delta


def _sse(obj) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _error(msg: str, code: int = 400) -> JSONResponse:
    return JSONResponse({"error": {"message": msg, "type": "invalid_request_error", "code": code}}, status_code=code)


def create_app(engine: AsyncEngine, model_name: str) -> FastAPI:
    app = FastAPI(title="tinyserve")
    max_len = engine.cfg.max_model_len

    def params_for(req, prompt_len: int, max_tokens: int | None) -> SamplingParams:
        limit = max_len - prompt_len
        if limit < 1:
            raise ValueError(f"prompt has {prompt_len} tokens; max_model_len is {max_len}")
        return SamplingParams(
            max_tokens=min(max_tokens or limit, limit), temperature=req.temperature, top_p=req.top_p,
            top_k=req.top_k, ignore_eos=req.ignore_eos, seed=req.seed, stop_token_ids=list(req.stop_token_ids),
        )

    async def run(prompt_ids, params, req, kind: str):
        rid = f"{'chatcmpl' if kind == 'chat' else 'cmpl'}-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        obj = "chat.completion" if kind == "chat" else "text_completion"
        detok = IncrementalDetokenizer(engine.tokenizer) if engine.tokenizer is not None else None
        gen = engine.generate(prompt_ids, params)

        def choice(text, finish, stream):
            if kind == "chat":
                key = "delta" if stream else "message"
                body = {"content": text} if stream else {"role": "assistant", "content": text}
                return {"index": 0, key: body, "finish_reason": finish, "logprobs": None}
            return {"index": 0, "text": text, "finish_reason": finish, "logprobs": None}

        def usage(p, c):
            return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}

        if not req.stream:
            text, all_ids, last = "", [], None
            async for out in gen:
                all_ids += out["token_ids"]
                last = out
            if detok is not None:
                text = detok.add(all_ids, final=True)
            return JSONResponse({"id": rid, "object": obj, "created": created, "model": model_name,
                                 "choices": [choice(text, last["finish_reason"], False)],
                                 "usage": usage(last["num_prompt_tokens"], last["num_completion_tokens"])})

        include_usage = bool(req.stream_options and req.stream_options.include_usage)

        async def stream():
            chunk_obj = "chat.completion.chunk" if kind == "chat" else "text_completion"
            base = {"id": rid, "object": chunk_obj, "created": created, "model": model_name}
            if kind == "chat":
                yield _sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
            last = None
            try:
                async for out in gen:
                    last = out
                    text = detok.add(out["token_ids"], final=out["finished"]) if detok else ""
                    if text or out["finished"]:
                        yield _sse({**base, "choices": [choice(text, out["finish_reason"], True)]})
            except Exception as e:  # noqa: BLE001
                yield _sse({"error": {"message": str(e)}})
                return
            if include_usage and last is not None:
                yield _sse({**base, "choices": [], "usage": usage(last["num_prompt_tokens"], last["num_completion_tokens"])})
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        try:
            if isinstance(req.prompt, str):
                if engine.tokenizer is None:
                    return _error("this server has no tokenizer; send token ids")
                prompt_ids = engine.tokenizer.encode(req.prompt)
            else:
                prompt_ids = list(req.prompt)
            if not prompt_ids:
                return _error("empty prompt")
            params = params_for(req, len(prompt_ids), req.max_tokens)
            if engine.cfg.spec_method and not params.greedy:
                return _error("this server runs speculative decoding, which is greedy-only: set temperature=0")
            return await run(prompt_ids, params, req, "completion")
        except ValueError as e:
            return _error(str(e))

    @app.post("/v1/chat/completions")
    async def chat(req: ChatCompletionRequest):
        if engine.tokenizer is None:
            return _error("this server has no tokenizer")
        try:
            prompt_ids = engine.tokenizer.apply_chat_template(
                [m.model_dump() for m in req.messages], tokenize=True, add_generation_prompt=True, **req.chat_template_kwargs)
            params = params_for(req, len(prompt_ids), req.max_completion_tokens or req.max_tokens)
            if engine.cfg.spec_method and not params.greedy:
                return _error("this server runs speculative decoding, which is greedy-only: set temperature=0")
            return await run(prompt_ids, params, req, "chat")
        except ValueError as e:
            return _error(str(e))

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "created": 0, "owned_by": "tinyserve",
                                            "max_model_len": max_len}]}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        return engine.metrics()

    @app.exception_handler(ValueError)
    async def value_error(_: Request, e: ValueError):
        return _error(str(e))

    return app


def main(argv=None) -> None:
    import uvicorn

    from tinyserve.config import EngineConfig
    from tinyserve.engine import LLMEngine

    ap = argparse.ArgumentParser(description="tinyserve OpenAI-compatible server")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--served-model-name", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn-backend", default="flash")
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-batched-tokens", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--num-kv-blocks", type=int, default=None)
    ap.add_argument("--no-prefix-cache", action="store_true")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--spec-method", default=None, choices=[None, "dflash"])
    ap.add_argument("--spec-draft-model", default=None)
    ap.add_argument("--engine-mode", default="process", choices=["process", "thread"],
                    help="process: engine core in its own process (default); thread: same process")
    a = ap.parse_args(argv)
    cfg = EngineConfig(model=a.model, device=a.device, dtype=a.dtype, attn_backend=a.attn_backend, block_size=a.block_size,
                       max_num_seqs=a.max_num_seqs, max_model_len=a.max_model_len, max_num_batched_tokens=a.max_num_batched_tokens,
                       gpu_memory_utilization=a.gpu_memory_utilization, num_kv_blocks=a.num_kv_blocks,
                       enable_prefix_cache=not a.no_prefix_cache, enforce_eager=a.enforce_eager,
                       spec_method=a.spec_method, spec_draft_model=a.spec_draft_model)
    if a.engine_mode == "process":
        from transformers import AutoTokenizer

        from tinyserve.loader import resolve_model_path
        from tinyserve.server.engine_process import ProcessAsyncEngine

        aeng = ProcessAsyncEngine(cfg, AutoTokenizer.from_pretrained(resolve_model_path(a.model)))
        n_blocks = aeng.num_kv_blocks
    else:
        engine = LLMEngine(cfg)
        aeng, n_blocks = AsyncEngine(engine), engine.block_manager.num_blocks
    app = create_app(aeng, a.served_model_name or a.model)
    print(f"tinyserve: serving {a.model} on http://{a.host}:{a.port} (engine {a.engine_mode}, KV blocks: {n_blocks})", flush=True)
    try:
        uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    finally:
        if hasattr(aeng, "shutdown"):
            aeng.shutdown()  # the engine process holds the GPU memory; never leave it behind


if __name__ == "__main__":
    main()
