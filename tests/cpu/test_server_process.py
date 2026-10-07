"""FIXPLAN D2: the server with the engine core in a separate process (CPU, tiny model)."""

import asyncio
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from tests.util import make_fullvocab_tiny_dir
from tinyserve.config import EngineConfig
from tinyserve.sampling import SamplingParams


@pytest.fixture(scope="module")
def proc_server(tmp_path_factory):
    from transformers import AutoTokenizer

    from tinyserve.server.api import create_app
    from tinyserve.server.engine_process import ProcessAsyncEngine

    path = make_fullvocab_tiny_dir(tmp_path_factory.mktemp("tiny_proc"))
    cfg = EngineConfig(model=str(path), device="cpu", dtype="float32", attn_backend="torch", block_size=16,
                       max_model_len=1024, max_num_batched_tokens=2048, max_num_seqs=8)
    aeng = ProcessAsyncEngine(cfg, AutoTokenizer.from_pretrained(path))
    yield create_app(aeng, "tiny"), aeng, path
    aeng.shutdown()


def _sse(text):
    ev = [line[6:] for line in text.split("\n\n") if line.startswith("data: ")]
    assert ev[-1] == "[DONE]"
    return [json.loads(e) for e in ev[:-1]]


def test_stream_equals_non_stream_and_thread_mode(proc_server):
    app, _, path = proc_server
    c = TestClient(app)
    req = {"prompt": "The quick brown fox", "max_tokens": 20, "temperature": 0, "ignore_eos": True}
    full = c.post("/v1/completions", json=req).json()
    assert full["usage"]["completion_tokens"] == 20
    ev = _sse(c.post("/v1/completions", json={**req, "stream": True, "stream_options": {"include_usage": True}}).text)
    assert "".join(e["choices"][0]["text"] for e in ev[:-1]) == full["choices"][0]["text"]
    # Same answer as the in-process engine (same weights, same float32 math).
    from tests.util import tiny_engine
    from transformers import AutoTokenizer

    eng = tiny_engine(path, dtype="float32", max_model_len=1024)
    tok = AutoTokenizer.from_pretrained(path)
    ref = eng.generate([tok.encode("The quick brown fox")], SamplingParams(max_tokens=20, ignore_eos=True))[0]["token_ids"]
    assert tok.decode(ref, skip_special_tokens=True) == full["choices"][0]["text"]


def test_concurrent_requests_and_metrics(proc_server):
    app, aeng, _ = proc_server

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            return await asyncio.gather(*[client.post("/v1/completions", json={
                "prompt": f"Request {i}", "max_tokens": 8 + i, "temperature": 0, "ignore_eos": True}) for i in range(8)])

    rs = asyncio.run(go())
    assert [r.json()["usage"]["completion_tokens"] for r in rs] == [8 + i for i in range(8)]
    m = TestClient(app).get("/metrics").json()
    assert m["running"] == 0 and m["finished"] >= 8


def test_abort_frees_the_sequence(proc_server):
    _, aeng, _ = proc_server

    async def go():
        gen = aeng.generate([1, 2, 3, 4], SamplingParams(max_tokens=500, ignore_eos=True))
        await gen.__anext__()  # first token arrived
        await gen.aclose()  # client disconnects

    asyncio.run(go())
    for _ in range(100):
        if aeng.metrics()["running"] == 0 and aeng.metrics()["kv_usage"] == 0:
            break
        time.sleep(0.05)
    assert aeng.metrics()["running"] == 0 and aeng.metrics()["kv_usage"] == 0


def test_bad_request_reports_error(proc_server):
    app, _, _ = proc_server
    assert TestClient(app).post("/v1/completions", json={"prompt": [5] * 2000, "max_tokens": 4}).status_code == 400
