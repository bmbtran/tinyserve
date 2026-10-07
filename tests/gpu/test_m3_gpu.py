"""M3 on an L4: batched == sequential (near-tie rule) and an offline throughput number."""

import time

import pytest
import torch

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
                                 enable_prefix_cache=False, enforce_eager=True))
    print(f"KV blocks: {eng.block_manager.num_blocks} x 256 tokens")
    return eng


def test_a_batched_vs_sequential(engine, report):
    rows = load_chat("mixed_prompts")
    prompts = [engine.apply_chat_template(r["messages"]) for r in rows]
    sp = SamplingParams(max_tokens=128)
    engine.runner.record_gaps = True
    seq_outs = [engine.generate([p], sp)[0] for p in prompts]  # bs = 1, one at a time
    engine.runner.record_gaps = False
    engine.cfg.max_num_seqs = 32
    t = time.perf_counter()
    batched = engine.generate(prompts, sp)
    report["a_batched_wall_s"] = round(time.perf_counter() - t, 2)
    engine.cfg.max_num_seqs = 128
    verdicts = [compare_tokens(s["token_ids"], b["token_ids"], s["logit_gaps"]) for s, b in zip(seq_outs, batched)]
    n_exact = sum(v[0] == "exact" for v in verdicts)
    n_ok = sum(v[0] != "mismatch" for v in verdicts)
    report["a_exact"] = f"{n_exact}/64"
    report["a_exact_or_near_tie"] = f"{n_ok}/64"
    report["a_divergences"] = [{"i": i, "verdict": v[0], "at": v[1], "gap": v[2]} for i, v in enumerate(verdicts) if v[0] != "exact"]
    print(f"M3a batched(32) vs sequential: exact={n_exact}/64 (>=60) exact_or_near_tie={n_ok}/64 (64)")
    assert n_exact >= 60 and n_ok == 64


def test_b_offline_throughput(engine, report):
    reqs = fixed_len(256, 512, 256, seed=0)
    sp = SamplingParams(max_tokens=256, ignore_eos=True)
    engine.generate([r["prompt"] for r in reqs[:8]], SamplingParams(max_tokens=8, ignore_eos=True))  # warmup
    before = dict(engine.counters["steps"])
    torch.cuda.synchronize()
    t = time.perf_counter()
    outs = engine.generate([r["prompt"] for r in reqs], sp)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t
    n_out = sum(len(o["token_ids"]) for o in outs)
    steps = {k: engine.counters["steps"][k] - before[k] for k in before}
    report["b_output_tok_s"] = round(n_out / wall, 1)
    report["b_total_tok_s"] = round((n_out + 256 * 512) / wall, 1)
    report["b_wall_s"] = round(wall, 2)
    report["b_steps"] = steps
    report["b_preemptions"] = engine.scheduler.num_preemptions
    print(f"M3b offline 256 x (in 512 / out 256), eager, max_num_seqs=128: {n_out / wall:.0f} output tok/s, "
          f"{(n_out + 256 * 512) / wall:.0f} total tok/s, wall {wall:.1f}s, steps {steps}")
    assert n_out == 256 * 256
