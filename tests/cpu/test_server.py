"""M6: OpenAI-compatible server on CPU (tiny random model + the real Qwen3 tokenizer)."""

import asyncio
import json
import shutil

import httpx
import pytest
from fastapi.testclient import TestClient

from tests.util import make_tiny_model, tiny_engine


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    from huggingface_hub import snapshot_download

    from tinyserve.server.api import create_app
    from tinyserve.server.async_engine import AsyncEngine

    path = tmp_path_factory.mktemp("tiny_qwen3_fullvocab")
    make_tiny_model(path, vocab_size=151936, eos_token_id=151645)
    tok_dir = snapshot_download("Qwen/Qwen3-0.6B", allow_patterns=["tokenizer*", "vocab.json", "merges.txt"])
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        shutil.copy(f"{tok_dir}/{f}", path / f)
    eng = tiny_engine(path, dtype="float32", max_model_len=1024)
    from transformers import AutoTokenizer

    eng.tokenizer = AutoTokenizer.from_pretrained(path)
    aeng = AsyncEngine(eng)
    app = create_app(aeng, "tiny")
    yield app, aeng
    aeng.shutdown()


def parse_sse(text: str) -> list:
    events = [line[len("data: "):] for line in text.split("\n\n") if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    return [json.loads(e) for e in events[:-1]]


def test_health_models_metrics(server):
    app, _ = server
    c = TestClient(app)
    assert c.get("/health").json() == {"status": "ok"}
    assert c.get("/v1/models").json()["data"][0]["id"] == "tiny"
    m = c.get("/metrics").json()
    assert {"running", "waiting", "kv_usage", "prefix_hit_rate", "spec_tau"} <= set(m)


def test_completion_non_stream(server):
    app, _ = server
    r = TestClient(app).post("/v1/completions", json={"prompt": "Hello there", "max_tokens": 12, "temperature": 0, "ignore_eos": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["usage"]["completion_tokens"] == 12
    assert body["choices"][0]["finish_reason"] == "length"
    assert isinstance(body["choices"][0]["text"], str)


def test_completion_token_ids_prompt(server):
    app, _ = server
    r = TestClient(app).post("/v1/completions", json={"prompt": [1, 2, 3, 4], "max_tokens": 5, "temperature": 0, "ignore_eos": True})
    assert r.json()["usage"] == {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9}


def test_stream_equals_non_stream_and_done(server):
    app, _ = server
    c = TestClient(app)
    req = {"prompt": "The quick brown fox", "max_tokens": 20, "temperature": 0, "ignore_eos": True}
    full = c.post("/v1/completions", json=req).json()["choices"][0]["text"]
    r = c.post("/v1/completions", json={**req, "stream": True, "stream_options": {"include_usage": True}})
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert events[-1]["usage"]["completion_tokens"] == 20 and events[-1]["choices"] == []
    streamed = "".join(e["choices"][0]["text"] for e in events[:-1])
    assert streamed == full
    assert events[-2]["choices"][0]["finish_reason"] == "length"


def test_chat_stream_equals_non_stream(server):
    app, _ = server
    c = TestClient(app)
    req = {"messages": [{"role": "user", "content": "Say hi"}], "max_tokens": 16, "temperature": 0, "ignore_eos": True}
    full = c.post("/v1/chat/completions", json=req).json()
    assert full["object"] == "chat.completion"
    text = full["choices"][0]["message"]["content"]
    events = parse_sse(c.post("/v1/chat/completions", json={**req, "stream": True}).text)
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    assert all(e["object"] == "chat.completion.chunk" for e in events)
    assert "".join(e["choices"][0]["delta"].get("content", "") for e in events) == text


def test_eight_concurrent_requests(server):
    app, aeng = server

    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            reqs = [client.post("/v1/completions", json={"prompt": f"Request number {i}", "max_tokens": 10 + i,
                                                          "temperature": 0, "ignore_eos": True, "stream": i % 2 == 0})
                    for i in range(8)]
            return await asyncio.gather(*reqs)

    rs = asyncio.run(go())
    assert all(r.status_code == 200 for r in rs)
    for i, r in enumerate(rs):
        if i % 2 == 0:
            assert r.text.rstrip().endswith("data: [DONE]")
        else:
            assert r.json()["usage"]["completion_tokens"] == 10 + i
    assert aeng.metrics()["running"] == 0


def test_bad_requests(server):
    app, _ = server
    c = TestClient(app)
    assert c.post("/v1/completions", json={"prompt": [], "max_tokens": 4}).status_code == 400
    assert c.post("/v1/completions", json={"prompt": [5] * 2000, "max_tokens": 4}).status_code == 400


def test_detokenizer_holds_back_partial_utf8():
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download

    from tinyserve.server.api import IncrementalDetokenizer

    tok = AutoTokenizer.from_pretrained(snapshot_download("Qwen/Qwen3-0.6B", allow_patterns=["tokenizer*", "vocab.json", "merges.txt"]))
    ids = tok.encode("naïve café 日本語 🙂 done")
    d = IncrementalDetokenizer(tok)
    pieces = [d.add([t], final=(i == len(ids) - 1)) for i, t in enumerate(ids)]
    assert "".join(pieces) == "naïve café 日本語 🙂 done"
    assert all("�" not in p for p in pieces)


def test_bench_client_against_server(server):
    """The benchmark client parses tinyserve's stream (completions + chat)."""
    from bench.client import run_load, summarize

    app, _ = server
    reqs = [{"prompt": [10 + i] * (5 + i), "max_tokens": 8, "ignore_eos": True} for i in range(6)]
    s = summarize(asyncio.run(run_load("http://t", reqs, 3, "tiny", transport=httpx.ASGITransport(app=app))))
    assert s["errors"] == 0 and s["output_tokens"] == 48 and s["ttft_ms"]["p50"] is not None
    chat = [{"messages": [{"role": "user", "content": f"hi {i}"}], "max_tokens": 6, "ignore_eos": True} for i in range(3)]
    s2 = summarize(asyncio.run(run_load("http://t", chat, 2, "tiny", endpoint="chat", transport=httpx.ASGITransport(app=app))))
    assert s2["errors"] == 0 and s2["output_tokens"] == 18
