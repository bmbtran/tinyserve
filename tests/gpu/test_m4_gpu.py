"""M4 on an L4: prefix cache on vs off (W2-style: 2048 shared + 64 unique tokens)."""

import statistics

import pytest

from bench.workloads import w2_shared_prefix
from tests.util import compare_tokens
from tinyserve.config import EngineConfig
from tinyserve.engine import LLMEngine
from tinyserve.sampling import SamplingParams

pytestmark = pytest.mark.gpu
SP = SamplingParams(max_tokens=64, ignore_eos=True)


@pytest.fixture(scope="module")
def engine():
    return LLMEngine(EngineConfig(model="Qwen/Qwen3-0.6B", dtype="bfloat16", device="cuda", attn_backend="flash",
                                  block_size=256, max_num_seqs=8, max_model_len=4096, max_num_batched_tokens=8192,
                                  enable_prefix_cache=True, enforce_eager=True))


def closed_loop(engine, prompts, concurrency):
    """First request alone, then keep `concurrency` in flight. TTFT is measured
    from when a request is added (so queueing inside this loop is excluded)."""
    first = engine.add_request(prompts[0], SP)
    while engine.has_work():
        engine.step()
    seqs, queue = [first], list(prompts[1:])
    live = []
    while queue or engine.has_work():
        while queue and len(live) < concurrency:
            live.append(engine.add_request(queue.pop(0), SP))
        _, finished = engine.step()
        for s in finished:
            live.remove(s)
            seqs.append(s)
    return seqs


def test_prefix_cache_on_vs_off(engine, report):
    reqs = w2_shared_prefix(groups=1, per_group=32, prefix_len=2048, suffix=(64, 64), out=64, seed=0)
    prompts = [r["prompt"] for r in reqs]
    engine.generate([prompts[0][:300]], SamplingParams(max_tokens=4, ignore_eos=True))  # warmup (unrelated tokens)

    engine.block_manager.enable_prefix_cache = False
    engine.runner.record_gaps = True
    off = closed_loop(engine, prompts, 8)
    engine.runner.record_gaps = False

    engine.block_manager.enable_prefix_cache = True
    engine.block_manager.stats = {"prefix_query_tokens": 0, "prefix_hit_tokens": 0}
    on = closed_loop(engine, prompts, 8)
    hit_rate = engine.block_manager.prefix_hit_rate

    off_by_prompt = {tuple(s.token_ids[: s.num_prompt_tokens]): s for s in off}
    verdicts = []
    for s in on:
        ref = off_by_prompt[tuple(s.token_ids[: s.num_prompt_tokens])]
        verdicts.append(compare_tokens(ref.completion_token_ids, s.completion_token_ids, ref.logit_gaps)[0])
    n_exact, n_ok = verdicts.count("exact"), sum(v != "mismatch" for v in verdicts)

    ttft = lambda seqs: statistics.median((s.first_token_time - s.arrival_time) * 1000 for s in seqs[1:])  # noqa: E731
    t_off, t_on = ttft(off), ttft(on)
    report.update(exact=f"{n_exact}/32", exact_or_near_tie=f"{n_ok}/32", hit_rate=round(hit_rate, 4),
                  ttft_p50_off_ms=round(t_off, 2), ttft_p50_on_ms=round(t_on, 2), ttft_ratio=round(t_on / t_off, 3),
                  ttft_speedup=round(t_off / t_on, 2))
    print(f"M4a cache on vs off: exact={n_exact}/32 (>=31) exact_or_near_tie={n_ok}/32 (32)")
    print(f"M4b hit_rate={hit_rate:.3f} (>=0.85)")
    print(f"M4c TTFT p50 off={t_off:.1f}ms on={t_on:.1f}ms ratio={t_on / t_off:.3f} (<=0.5) ttft_speedup={t_off / t_on:.2f}x")
    assert hit_rate >= 0.85
    assert t_on <= 0.5 * t_off
    assert n_exact >= 31 and n_ok == 32
