"""M9(c) on an L4: CUDA graphs for the DFlash verify step (Qwen3-4B + z-lab draft).

Same engine, graphs on vs off (off = runner.spec_graphs emptied): outputs
(near-tie rule), tau, bs=1 decode rate and c=16 throughput on GSM8K.
"""

import statistics
from contextlib import contextmanager

import pytest

from bench.workloads import load_chat
from tests.util import closed_loop, compare_tokens
from tinyserve.config import EngineConfig
from tinyserve.engine import LLMEngine
from tinyserve.sampling import SamplingParams

pytestmark = pytest.mark.gpu
SP = SamplingParams(max_tokens=512)


def rate(seqs):
    return statistics.fmean((s.num_completion_tokens - 1) / (s.finish_time - s.first_token_time) for s in seqs)


def tau(seqs):
    st = sum(s.spec_steps for s in seqs)
    return (sum(s.spec_accepted for s in seqs) + st) / st


@contextmanager
def eager(eng):
    saved = eng.runner.spec_graphs
    eng.runner.spec_graphs = {}
    try:
        yield
    finally:
        eng.runner.spec_graphs = saved


def bs1(eng, prompts):
    out = []
    for p in prompts:
        s = eng.add_request(p, SP)
        while eng.has_work():
            eng.step()
        out.append(s)
    return out


def test_spec_verify_graphs(report):
    eng = LLMEngine(EngineConfig(model="Qwen/Qwen3-4B", dtype="bfloat16", device="cuda", attn_backend="flash", block_size=256,
                                 max_num_seqs=16, max_model_len=4096, max_num_batched_tokens=8192, spec_method="dflash",
                                 spec_draft_model="z-lab/Qwen3-4B-DFlash-b16", cuda_graph_batch_sizes=(1, 2, 4, 8, 16)))
    print(f"spec verify graphs captured for {sorted(eng.runner.spec_graphs)}")
    prompts = [eng.apply_chat_template(r["messages"]) for r in load_chat("gsm8k")]
    eng.generate([prompts[0][:64]], SamplingParams(max_tokens=32))  # warmup

    eng.runner.record_gaps = True
    with eager(eng):
        e1 = bs1(eng, prompts[:8])
    eng.runner.record_gaps = False
    g1 = bs1(eng, prompts[:8])
    with eager(eng):
        e16, we = closed_loop(eng, prompts, 16, SP)
    g16, wg = closed_loop(eng, prompts, 16, SP)

    v = [compare_tokens(a.completion_token_ids, b.completion_token_ids, a.logit_gaps)[0] for a, b in zip(e1, g1)]
    r = {
        "bs1_eager_tok_s": round(rate(e1), 1), "bs1_graph_tok_s": round(rate(g1), 1),
        "bs1_speedup_graph_vs_eager": round(rate(g1) / rate(e1), 3),
        "c16_eager_tok_s": round(sum(s.num_completion_tokens for s in e16) / we, 1),
        "c16_graph_tok_s": round(sum(s.num_completion_tokens for s in g16) / wg, 1),
        "tau_bs1_eager": round(tau(e1), 3), "tau_bs1_graph": round(tau(g1), 3),
        "tau_c16_eager": round(tau(e16), 3), "tau_c16_graph": round(tau(g16), 3),
        "outputs_bs1_exact": f"{v.count('exact')}/8", "outputs_bs1_exact_or_near_tie": f"{sum(x != 'mismatch' for x in v)}/8",
    }
    report.update(r)
    print(f"M9c bs=1: eager {r['bs1_eager_tok_s']} tok/s -> graph {r['bs1_graph_tok_s']} tok/s ({r['bs1_speedup_graph_vs_eager']}x)")
    print(f"M9c c=16: eager {r['c16_eager_tok_s']} tok/s -> graph {r['c16_graph_tok_s']} tok/s")
    print(f"M9c tau bs1 {r['tau_bs1_eager']} -> {r['tau_bs1_graph']}; c16 {r['tau_c16_eager']} -> {r['tau_c16_graph']}; "
          f"outputs exact {r['outputs_bs1_exact']}, exact-or-near-tie {r['outputs_bs1_exact_or_near_tie']}")
    del eng
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    base = LLMEngine(EngineConfig(model="Qwen/Qwen3-4B", dtype="bfloat16", device="cuda", attn_backend="flash", block_size=256,
                                  max_num_seqs=16, max_model_len=4096, max_num_batched_tokens=8192,
                                  cuda_graph_batch_sizes=(1, 2, 4, 8, 16)))
    base.generate([prompts[0][:64]], SamplingParams(max_tokens=32))
    b1 = rate(bs1(base, prompts[:8]))
    report["bs1_nonspec_graph_tok_s"] = round(b1, 1)
    report["bs1_dflash_graph_speedup_vs_nonspec"] = round(rate(g1) / b1, 3)
    print(f"M9c bs=1 DFlash(graph verify) {rate(g1):.1f} vs non-spec (graphs) {b1:.1f} tok/s -> {rate(g1) / b1:.2f}x")
    assert sum(x != "mismatch" for x in v) == 8
    assert rate(g1) > rate(e1)
