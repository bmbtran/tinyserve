"""M6 on an L4: server + official `openai` client + bench client in one container."""

import asyncio
import subprocess
import sys

import pytest

from bench.client import run_load, summarize, wait_for_health
from bench.workloads import fixed_len

pytestmark = pytest.mark.gpu
PORT = 8011
BASE = f"http://127.0.0.1:{PORT}"
MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def server():
    proc = subprocess.Popen([sys.executable, "-m", "tinyserve.server.api", "--model", MODEL, "--port", str(PORT)],
                            stdout=sys.stdout, stderr=sys.stderr)
    try:
        waited = wait_for_health(BASE, 900, proc)
        print(f"server healthy after {waited:.0f}s")
        yield proc
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def test_openai_client_streaming(server, report):
    from openai import OpenAI

    client = OpenAI(base_url=f"{BASE}/v1", api_key="none")
    transcript = []
    stream = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "Name three primary colors."}], max_tokens=48,
        temperature=0, stream=True, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    chat_text = "".join(ch.choices[0].delta.content or "" for ch in stream if ch.choices)
    transcript.append(f"chat (stream) -> {chat_text!r}")
    comp = client.completions.create(model=MODEL, prompt="The capital of France is", max_tokens=12, temperature=0, stream=True)
    comp_text = "".join(ch.choices[0].text for ch in comp if ch.choices)
    transcript.append(f"completions (stream) -> {comp_text!r}")
    full = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Name three primary colors."}],
                                          max_tokens=48, temperature=0,
                                          extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    transcript.append(f"chat (non-stream) -> {full.choices[0].message.content!r} usage={full.usage}")
    models = client.models.list()
    transcript.append(f"models -> {[m.id for m in models.data]}")
    for t in transcript:
        print("  openai:", t)
    report["openai_transcript"] = transcript
    assert chat_text and comp_text
    assert full.choices[0].message.content == chat_text


def test_concurrency_32_x_64(server, report):
    reqs = [{**r, "ignore_eos": True} for r in fixed_len(64, 256, 128, seed=3)]
    asyncio.run(run_load(BASE, reqs[:8], 8, MODEL))  # warmup
    s = summarize(asyncio.run(run_load(BASE, reqs, 32, MODEL)))
    report["load"] = s
    report["errors"] = s["errors"]
    print(f"M6 c=32 x 64 req: errors={s['errors']} TTFT p50={s['ttft_ms']['p50']}ms p99={s['ttft_ms']['p99']}ms "
          f"ITL p50={s['itl_ms']['p50']}ms p99={s['itl_ms']['p99']}ms output_tok_s={s['output_tok_s']}")
    assert s["errors"] == 0
