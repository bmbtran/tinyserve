"""M5 on an L4: CUDA-graph decode vs eager (same engine; eager = graphs disabled)."""

import statistics
from contextlib import contextmanager

import pytest

from bench.workloads import fixed_len, load_chat
from tests.util import compare_tokens
from tinyserve.config import EngineConfig
from tinyserve.engine import LLMEngine
from tinyserve.sampling import SamplingParams

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def engine():
    eng = LLMEngine(EngineConfig(model="Qwen/Qwen3-0.6B", dtype="bfloat16", device="cuda", attn_backend="flash",
                                 block_size=256, max_num_seqs=128, max_model_len=4096, max_num_batched_tokens=8192,
                                 enable_prefix_cache=False, enforce_eager=False))
    print(f"captured graphs for batch sizes {sorted(eng.runner.graphs)}")
    return eng


@contextmanager
def eager(engine):
    saved = engine.runner.graphs
    engine.runner.graphs = {}
    try:
        yield
    finally:
        engine.runner.graphs = saved


def decode_itl_ms(engine, bs, out=64):
    """Median gap between consecutive tokens of the same sequence (pure decode)."""
    outs = engine.generate([r["prompt"] for r in fixed_len(bs, 64, out, seed=bs)], SamplingParams(max_tokens=out, ignore_eos=True))
    gaps = [b - a for o in outs for a, b in zip(o["token_times"][1:], o["token_times"][2:])]
    return statistics.median(gaps) * 1000


def test_a_graph_vs_eager_outputs(engine, report):
    prompts = [engine.apply_chat_template(r["messages"]) for r in load_chat("mixed_prompts", 32)]
    sp = SamplingParams(max_tokens=128)
    engine.runner.record_gaps = True
    with eager(engine):
        ref = engine.generate(prompts, sp)
    engine.runner.record_gaps = False
    got = engine.generate(prompts, sp)
    v = [compare_tokens(r["token_ids"], g["token_ids"], r["logit_gaps"]) for r, g in zip(ref, got)]
    n_exact, n_ok = sum(x[0] == "exact" for x in v), sum(x[0] != "mismatch" for x in v)
    report.update(a_exact=f"{n_exact}/32", a_exact_or_near_tie=f"{n_ok}/32",
                  a_divergences=[{"i": i, "verdict": x[0], "at": x[1], "gap": x[2]} for i, x in enumerate(v) if x[0] != "exact"])
    print(f"M5a graph vs eager: exact={n_exact}/32 (>=31) exact_or_near_tie={n_ok}/32 (32)")
    assert n_exact >= 31 and n_ok == 32


def test_b_c_itl(engine, report):
    decode_itl_ms(engine, 4, 16)  # warmup
    sweep = {}
    for bs in (1, 2, 4, 8, 16, 32, 64, 128):
        g = decode_itl_ms(engine, bs)
        with eager(engine):
            e = decode_itl_ms(engine, bs)
        sweep[bs] = {"graph_ms": round(g, 3), "eager_ms": round(e, 3), "speedup": round(e / g, 2)}
        print(f"  bs={bs:3d}: graph {g:.2f} ms  eager {e:.2f} ms  speedup {e / g:.2f}x")
    report["itl_sweep"] = sweep
    r1 = sweep[1]["graph_ms"] / sweep[1]["eager_ms"]
    report["b_bs1_graph_eager_ratio"] = round(r1, 3)
    report["c_bs32_graph_ms"], report["c_bs32_eager_ms"] = sweep[32]["graph_ms"], sweep[32]["eager_ms"]
    print(f"M5b bs=1 ITL p50 graph/eager={r1:.3f} (<=0.67)")
    print(f"M5c bs=32 step graph={sweep[32]['graph_ms']}ms eager={sweep[32]['eager_ms']}ms (graph<=eager)")
    assert r1 <= 0.67
    assert sweep[32]["graph_ms"] <= sweep[32]["eager_ms"]
