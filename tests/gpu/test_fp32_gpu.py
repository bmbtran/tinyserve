"""FIXPLAN B1: float32 exactness on the GPU (Qwen3-0.6B, torch attention backend).

Replaces the bf16 exact-match targets of M2c / M2a / M3a / M4a, which measured
bf16 rounding. Thresholds were fixed in FIXPLAN.md before this ran.
fp32-near-tie = the reference's top-1/top-2 gap < 0.01.
"""

import gc

import pytest
import torch

from bench.workloads import load_chat, w2_shared_prefix
from tests.util import closed_loop, compare_tokens, hf_greedy
from tinyserve.attention import AttnMetadata, backend_flash, backend_torch, set_attn_metadata
from tinyserve.config import EngineConfig
from tinyserve.engine import LLMEngine
from tinyserve.sampling import SamplingParams

pytestmark = pytest.mark.gpu
MODEL = "Qwen/Qwen3-0.6B"
FP32_TIE = 0.01
PROMPTS = [
    "The capital of France is", "Write a Python function that checks whether a number is prime.",
    "Explain the difference between TCP and UDP.", "Once upon a time, in a small village by the sea,",
    "List five uses of a paperclip:", "Translate to German: The weather is nice today.",
    "What is 17 times 23? Show your work.", "Summarize the plot of Romeo and Juliet in two sentences.",
]


def engine(dtype, backend, **kw):
    base = dict(model=MODEL, dtype=dtype, device="cuda", attn_backend=backend, block_size=256, num_kv_blocks=96,
                max_num_seqs=32, max_model_len=4096, max_num_batched_tokens=8192, enforce_eager=True)
    base.update(kw)
    return LLMEngine(EngineConfig(**base))


def prefill_logits(model, seqs):
    lens = [len(s) for s in seqs]
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda")
    set_attn_metadata(AttnMetadata(is_varlen=True, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=max(lens),
                                   max_seqlen_k=max(lens), q_lens=lens, k_lens=lens))
    try:
        with torch.inference_mode():
            h, _ = model(torch.tensor(sum(seqs, []), device="cuda"), torch.cat([torch.arange(n, device="cuda") for n in lens]))
            return model.compute_logits(h).float()
    finally:
        set_attn_metadata(None)


def verdicts(refs, gots, gaps):
    v = [compare_tokens(r, g, gp, thr=FP32_TIE) for r, g, gp in zip(refs, gots, gaps)]
    return v, sum(x[0] == "exact" for x in v), sum(x[0] != "mismatch" for x in v)


def test_b1a_b1b_vs_hf_fp32(report):
    from transformers import AutoModelForCausalLM

    assert not torch.backends.cuda.matmul.allow_tf32  # true fp32 GEMMs
    e32 = engine("float32", "torch")
    tok = e32.tokenizer
    prompts = [tok.encode(p) for p in PROMPTS]
    hf = AutoModelForCausalLM.from_pretrained(e32.model_path, dtype=torch.float32, attn_implementation="sdpa").cuda().eval()

    # B1b part 1: fp32 prefill logits vs HF fp32
    seqs = [tok.encode(p * 6) for p in PROMPTS[:4]]
    l32 = prefill_logits(e32.runner.model, seqs)
    with torch.inference_mode():
        lhf = torch.cat([hf(torch.tensor([s], device="cuda")).logits[0].float() for s in seqs])
    d = (l32 - lhf).abs().max().item()
    report["b1b_fp32_vs_hf_max_abs"] = d
    print(f"B1b fp32 prefill logits vs HF fp32: max_abs={d:.2e} (<1e-3)")

    # B1a: greedy 128 tokens vs HF fp32
    refs = [hf_greedy(hf, p, 128, return_gaps=True) for p in prompts]
    del hf
    gc.collect()
    torch.cuda.empty_cache()
    outs = e32.generate(prompts, SamplingParams(max_tokens=128, ignore_eos=True))
    v, n_exact, n_ok = verdicts([r[0] for r in refs], [o["token_ids"] for o in outs], [r[1] for r in refs])
    report["b1a_exact"], report["b1a_exact_or_fp32_near_tie"] = f"{n_exact}/8", f"{n_ok}/8"
    report["b1a_divergences"] = [{"i": i, "verdict": x[0], "at": x[1], "gap": x[2]} for i, x in enumerate(v) if x[0] != "exact"]
    print(f"B1a fp32 greedy vs HF fp32: exact={n_exact}/8 (>=7) exact_or_fp32_near_tie={n_ok}/8 (8)")

    # B1b part 2: the bf16 kernel check restated against fp32 ground truth (post-hoc 1.25x criterion)
    e16 = engine("bfloat16", "flash", num_kv_blocks=8)
    lf = prefill_logits(e16.runner.model, seqs)
    for layer in e16.runner.layers:
        layer.backend = backend_torch
    lt = prefill_logits(e16.runner.model, seqs)
    for layer in e16.runner.layers:
        layer.backend = backend_flash
    err_f, err_t = (lf - l32).abs().max().item(), (lt - l32).abs().max().item()
    top1_f = (lf.argmax(-1) == l32.argmax(-1)).float().mean().item()
    report.update(b1b_flash_bf16_err_vs_fp32=round(err_f, 4), b1b_torch_bf16_err_vs_fp32=round(err_t, 4),
                  b1b_flash_err_ratio=round(err_f / err_t, 3), b1b_flash_top1_vs_fp32=round(top1_f, 4))
    print(f"B1b bf16 vs fp32 truth: flash err={err_f:.3f} torch-bf16 err={err_t:.3f} ratio={err_f / err_t:.2f} (<=1.25) "
          f"flash top1={top1_f:.4f} (>=0.99)")
    del e16, e32
    gc.collect()
    torch.cuda.empty_cache()
    assert d < 1e-3
    assert n_exact >= 7 and n_ok == 8
    assert err_f <= 1.25 * err_t and top1_f >= 0.99


def test_b1c_batched_vs_sequential_fp32(report):
    e32 = engine("float32", "torch")
    prompts = [e32.apply_chat_template(r["messages"]) for r in load_chat("mixed_prompts")]
    sp = SamplingParams(max_tokens=128)
    e32.runner.record_gaps = True
    seq_outs = [e32.generate([p], sp)[0] for p in prompts]
    e32.runner.record_gaps = False
    batched = e32.generate(prompts, sp)  # max_num_seqs=32
    v, n_exact, n_ok = verdicts([s["token_ids"] for s in seq_outs], [b["token_ids"] for b in batched], [s["logit_gaps"] for s in seq_outs])
    report["b1c_exact"], report["b1c_exact_or_fp32_near_tie"] = f"{n_exact}/64", f"{n_ok}/64"
    report["b1c_divergences"] = [{"i": i, "verdict": x[0], "at": x[1], "gap": x[2]} for i, x in enumerate(v) if x[0] != "exact"]
    print(f"B1c fp32 batched(32) vs sequential: exact={n_exact}/64 (>=60) exact_or_fp32_near_tie={n_ok}/64 (64)")
    del e32
    gc.collect()
    torch.cuda.empty_cache()
    assert n_exact >= 60 and n_ok == 64


def test_b1d_prefix_cache_fp32(report):
    e32 = engine("float32", "torch", max_num_seqs=8)
    prompts = [r["prompt"] for r in w2_shared_prefix(groups=1, per_group=32, prefix_len=2048, suffix=(64, 64), out=64, seed=0)]
    sp = SamplingParams(max_tokens=64, ignore_eos=True)
    e32.block_manager.enable_prefix_cache = False
    e32.runner.record_gaps = True
    off, _ = closed_loop(e32, prompts, 8, sp, first_alone=True)
    e32.runner.record_gaps = False
    e32.block_manager.enable_prefix_cache = True
    e32.block_manager.stats = {"prefix_query_tokens": 0, "prefix_hit_tokens": 0}
    on, _ = closed_loop(e32, prompts, 8, sp, first_alone=True)
    v, n_exact, n_ok = verdicts([s.completion_token_ids for s in off], [s.completion_token_ids for s in on], [s.logit_gaps for s in off])
    hit = e32.block_manager.prefix_hit_rate
    report["b1d_exact"], report["b1d_exact_or_fp32_near_tie"], report["b1d_hit_rate"] = f"{n_exact}/32", f"{n_ok}/32", round(hit, 4)
    report["b1d_divergences"] = [{"i": i, "verdict": x[0], "at": x[1], "gap": x[2]} for i, x in enumerate(v) if x[0] != "exact"]
    print(f"B1d fp32 prefix cache on vs off: exact={n_exact}/32 (>=31) exact_or_fp32_near_tie={n_ok}/32 (32) hit_rate={hit:.3f}")
    assert hit > 0.85
    assert n_exact >= 31 and n_ok == 32
