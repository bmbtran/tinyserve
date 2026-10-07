"""FIXPLAN A1: settle the M7c bf16 divergences of DFlash spec vs non-spec (Qwen3-4B).

Phase 1 (bf16, production code: flash-attn, CUDA graphs incl. the spec verify step):
  spec vs non-spec on all 96 W3 prompts, saving prompt indices, tokens and gaps.
Phase 2 (float32 target AND draft, torch backend): for every bf16 mismatch plus the
  first 8 near-ties, run spec and non-spec again in float32 and compare; also get the
  float32 model's own top-1/top-2 at each bf16 divergence point.
PASS iff every fp32 pair is exact or diverges only at an fp32 gap < 0.01, and every
bf16 mismatch is exact in fp32 (or there are no bf16 mismatches).
"""

import gc

import pytest
import torch

from bench.workloads import SPEC_DATASETS, load_chat
from tests.util import closed_loop, compare_tokens
from tinyserve.attention import AttnMetadata, set_attn_metadata
from tinyserve.config import EngineConfig
from tinyserve.engine import LLMEngine
from tinyserve.sampling import SamplingParams

pytestmark = pytest.mark.gpu
TARGET, DRAFT = "Qwen/Qwen3-4B", "z-lab/Qwen3-4B-DFlash-b16"
SP = SamplingParams(max_tokens=512)
FP32_TIE = 0.01
S: dict = {}


def make(dtype, spec, **kw):
    base = dict(model=TARGET, dtype=dtype, device="cuda", attn_backend="flash" if dtype == "bfloat16" else "torch",
                block_size=256, max_num_seqs=16, max_model_len=4096, max_num_batched_tokens=8192,
                cuda_graph_batch_sizes=(1, 2, 4, 8, 16), enforce_eager=dtype != "bfloat16")
    if spec:
        base.update(spec_method="dflash", spec_draft_model=DRAFT)
    base.update(kw)
    return LLMEngine(EngineConfig(**base))


def free():
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def top2_at(model, ids):
    """The model's own top-1/top-2 for the token after `ids` (one fp32 prefill)."""
    n = len(ids)
    cu = torch.tensor([0, n], dtype=torch.int32, device="cuda")
    set_attn_metadata(AttnMetadata(is_varlen=True, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=n, max_seqlen_k=n,
                                   q_lens=[n], k_lens=[n]))
    try:
        h, _ = model(torch.tensor(ids, device="cuda"), torch.arange(n, device="cuda"))
        v, i = model.compute_logits(h[-1:]).float().topk(2)
    finally:
        set_attn_metadata(None)
    return i[0].tolist(), (v[0, 0] - v[0, 1]).item()


@pytest.fixture(scope="module")
def prompts():
    from transformers import AutoTokenizer

    from tinyserve.loader import resolve_model_path

    tok = AutoTokenizer.from_pretrained(resolve_model_path(TARGET))
    rows = [(d, i, r["messages"]) for d in SPEC_DATASETS for i, r in enumerate(load_chat(d))]
    return [(d, i, tok.apply_chat_template(m, tokenize=True, add_generation_prompt=True, enable_thinking=False)) for d, i, m in rows]


def test_1_bf16_production(prompts, report):
    ids = [p for _, _, p in prompts]
    eng = make("bfloat16", spec=True)
    assert eng.runner.spec_graphs, "expected CUDA graphs for the verify step"
    spec, _ = closed_loop(eng, ids, 16, SP)
    S["spec16"] = [s.completion_token_ids for s in spec]
    del eng, spec
    free()
    eng = make("bfloat16", spec=False)
    eng.runner.record_gaps = True
    base, _ = closed_loop(eng, ids, 16, SP)
    S["base16"] = [(s.completion_token_ids, s.logit_gaps) for s in base]
    del eng, base
    free()
    rows = []
    for k, ((d, i, _), sp, (bt, gp)) in enumerate(zip(prompts, S["spec16"], S["base16"])):
        v, at, gap = compare_tokens(bt, sp, gp)
        rows.append({"k": k, "dataset": d, "idx": i, "verdict": v, "at": at, "gap": gap,
                     "spec_tok": sp[at] if at is not None and at < len(sp) else None,
                     "base_tok": bt[at] if at is not None and at < len(bt) else None})
    S["rows"] = rows
    n_ex = sum(r["verdict"] == "exact" for r in rows)
    n_ok = sum(r["verdict"] != "mismatch" for r in rows)
    report["bf16_exact"], report["bf16_exact_or_near_tie"] = f"{n_ex}/96", f"{n_ok}/96"
    print(f"A1 bf16 (graph verify) spec vs non-spec: exact={n_ex}/96 exact_or_near_tie={n_ok}/96")
    for r in rows:
        if r["verdict"] == "mismatch":
            print(f"  bf16 MISMATCH {r['dataset']}#{r['idx']} at token {r['at']} gap={r['gap']}")


def test_2_fp32(prompts, report):
    rows = S["rows"]
    picked = [r for r in rows if r["verdict"] == "mismatch"] + [r for r in rows if r["verdict"] == "near_tie"][:8]
    ids = [prompts[r["k"]][2] for r in picked]
    print(f"A1 fp32 phase on {len(picked)} prompts ({sum(r['verdict'] == 'mismatch' for r in picked)} bf16 mismatches)")

    base32 = make("float32", spec=False, num_kv_blocks=20, max_num_seqs=4)
    # fp32 truth at each bf16 divergence point
    for r in picked:
        prefix = prompts[r["k"]][2] + S["base16"][r["k"]][0][: r["at"]]
        (t1, t2), gap = top2_at(base32.runner.model, prefix)
        r["fp32_top2"], r["fp32_gap"] = [t1, t2], gap
        r["bf16_choices_are_fp32_top2"] = {r["spec_tok"], r["base_tok"]} <= {t1, t2}
    base32.runner.record_gaps = True
    b, _ = closed_loop(base32, ids, 4, SP)
    b = [(s.completion_token_ids, s.logit_gaps) for s in b]
    del base32
    free()
    spec32 = make("float32", spec=True, num_kv_blocks=16, max_num_seqs=4)
    s, _ = closed_loop(spec32, ids, 4, SP)
    s = [x.completion_token_ids for x in s]
    del spec32
    free()

    all_ok, mism_ok = True, True
    for r, (bt, gp), st in zip(picked, b, s):
        v, at, gap = compare_tokens(bt, st, gp, thr=FP32_TIE)
        r["fp32_verdict"], r["fp32_at"], r["fp32_div_gap"] = v, at, gap
        all_ok &= v != "mismatch"
        if r["verdict"] == "mismatch":
            mism_ok &= v == "exact"
        print(f"  {r['dataset']}#{r['idx']}: bf16 {r['verdict']} at {r['at']} (gap {r['gap']}) | fp32 truth gap there "
              f"{r['fp32_gap']:.4f}, bf16 picks in fp32 top-2: {r['bf16_choices_are_fp32_top2']} | fp32 spec vs non-spec: {v}"
              + (f" at {at} gap {gap:.5f}" if at is not None else ""))
    report["picked"] = picked
    report["fp32_pairs_exact"] = f"{sum(r['fp32_verdict'] == 'exact' for r in picked)}/{len(picked)}"
    report["fp32_pairs_ok"] = f"{sum(r['fp32_verdict'] != 'mismatch' for r in picked)}/{len(picked)}"
    report["bf16_mismatches_exact_in_fp32"] = mism_ok
    print(f"A1 fp32 spec vs non-spec: exact={report['fp32_pairs_exact']} ok={report['fp32_pairs_ok']}; "
          f"every bf16 mismatch exact in fp32: {mism_ok}")
    assert all_ok and mism_ok
