"""M7 on an L4: DFlash with Qwen3-4B + z-lab/Qwen3-4B-DFlash-b16 (bf16).

Phases run in order in ONE container, freeing GPU memory in between:
  1. reference: vendored z-lab spec_generate (HF target, sdpa) on 8 prompts per dataset
  2. tinyserve DFlash: tau at bs=1 (same 8 prompts), tau on all 32 per dataset, speed at c=1/4/16
  3. tinyserve non-spec (CUDA graphs): outputs for the lossless check, speed at c=1/4/16
  4. checks: (a) tau parity +-10%, (b) tau >= 0.7x paper, (c) lossless, (d) speedup >= 2x at bs=1
"""

import gc
import statistics

import pytest
import torch

from bench.workloads import PAPER_TAU_QWEN3_4B, SPEC_DATASETS, load_chat
from tests.util import closed_loop, compare_tokens

pytestmark = pytest.mark.gpu
TARGET, DRAFT = "Qwen/Qwen3-4B", "z-lab/Qwen3-4B-DFlash-b16"
MAX_TOKENS = 512
N_REF = 8
PAPER_MIN = {"gsm8k": 4.5, "humaneval": 4.6, "mtbench": 3.0}  # PLAN.md M7(b)
S: dict = {}  # shared state between the ordered tests


def free():
    gc.collect()
    torch.cuda.empty_cache()


def pooled_tau(seqs):
    steps = sum(s.spec_steps for s in seqs)
    return (sum(s.spec_accepted for s in seqs) + steps) / steps


def decode_rate(seqs):
    """Mean per-request decode tokens/s (excludes prefill / TTFT)."""
    return statistics.fmean((s.num_completion_tokens - 1) / (s.finish_time - s.first_token_time)
                            for s in seqs if s.num_completion_tokens > 1)


def make_engine(spec: bool, **kw):
    from tinyserve.config import EngineConfig
    from tinyserve.engine import LLMEngine

    base = dict(model=TARGET, dtype="bfloat16", device="cuda", attn_backend="flash", block_size=256, max_num_seqs=16,
                max_model_len=4096, max_num_batched_tokens=8192, enable_prefix_cache=True,
                cuda_graph_batch_sizes=(1, 2, 4, 8, 16))
    if spec:
        base.update(spec_method="dflash", spec_draft_model=DRAFT)
    base.update(kw)
    return LLMEngine(EngineConfig(**base))


@pytest.fixture(scope="module")
def prompts():
    from transformers import AutoTokenizer

    from tinyserve.loader import resolve_model_path

    path = resolve_model_path(TARGET)
    tok = AutoTokenizer.from_pretrained(path)
    S["eos"] = [151645, 151643]
    return {d: [tok.apply_chat_template(r["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False)
                for r in load_chat(d)] for d in SPEC_DATASETS}


def test_1_reference(prompts, report):
    from transformers import AutoModelForCausalLM

    from tests.reference.dflash_ref import DFlashDraftModel
    from tinyserve.loader import resolve_model_path

    target = AutoModelForCausalLM.from_pretrained(resolve_model_path(TARGET), dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    draft = DFlashDraftModel.from_pretrained(resolve_model_path(DRAFT), dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    ref = {}
    for d in SPEC_DATASETS:
        acc_all, outs = [], []
        for p in prompts[d][:N_REF]:
            out, acc, _ = draft.spec_generate(target=target, input_ids=torch.tensor([p], device="cuda"), max_new_tokens=MAX_TOKENS,
                                              stop_token_ids=S["eos"], temperature=0.0, return_stats=True)
            acc_all += acc
            outs.append(out[0, len(p):].tolist())
        ref[d] = {"tau": sum(acc_all) / len(acc_all), "steps": len(acc_all)}
        S.setdefault("ref_outputs", {})[d] = outs
        print(f"  reference {d}: tau={ref[d]['tau']:.3f} over {len(acc_all)} steps")
    S["ref"] = ref
    report["ref_tau"] = {d: round(v["tau"], 3) for d, v in ref.items()}
    del target, draft
    free()


def test_2_tinyserve_spec(prompts, report):
    from tinyserve.sampling import SamplingParams

    eng = make_engine(spec=True)
    sp = SamplingParams(max_tokens=MAX_TOKENS)
    eng.generate([prompts["gsm8k"][0][:64]], SamplingParams(max_tokens=16))  # warmup
    bs1 = {}
    for d in SPEC_DATASETS:  # (a) bs=1, one request at a time
        seqs = []
        for p in prompts[d][:N_REF]:
            s = eng.add_request(p, sp)
            while eng.has_work():
                eng.step()
            seqs.append(s)
        bs1[d] = seqs
        print(f"  tinyserve bs=1 {d}: tau={pooled_tau(seqs):.3f} decode {decode_rate(seqs):.1f} tok/s")
    S["ts_tau_bs1"] = {d: pooled_tau(v) for d, v in bs1.items()}
    S["spec_rate"] = {1: decode_rate(bs1["gsm8k"])}
    S["spec_tokens_bs1"] = {d: [s.completion_token_ids for s in v] for d, v in bs1.items()}

    all_out = {}
    for d in SPEC_DATASETS:  # (b) all 32 prompts, concurrency 16
        seqs, _ = closed_loop(eng, prompts[d], 16, sp)
        all_out[d] = seqs
        print(f"  tinyserve c=16 {d}: tau={pooled_tau(seqs):.3f}")
    S["ts_tau_all"] = {d: pooled_tau(v) for d, v in all_out.items()}
    S["spec_outputs"] = {d: [s.completion_token_ids for s in v] for d, v in all_out.items()}

    for c in (4, 16):  # (d) throughput on gsm8k
        seqs, wall = closed_loop(eng, prompts["gsm8k"], c, sp)
        S["spec_rate"][c] = sum(s.num_completion_tokens for s in seqs) / wall
        print(f"  tinyserve DFlash gsm8k c={c}: {S['spec_rate'][c]:.1f} output tok/s")
    report["tinyserve_tau_bs1_first8"] = {d: round(v, 3) for d, v in S["ts_tau_bs1"].items()}
    report["tinyserve_tau_all32_c16"] = {d: round(v, 3) for d, v in S["ts_tau_all"].items()}
    report["engine_metrics_spec"] = eng.metrics()
    del eng
    free()


def test_3_tinyserve_nonspec(prompts, report):
    from tinyserve.sampling import SamplingParams

    eng = make_engine(spec=False)
    print(f"  non-spec engine graphs: {sorted(eng.runner.graphs)}")
    sp = SamplingParams(max_tokens=MAX_TOKENS)
    eng.generate([prompts["gsm8k"][0][:64]], SamplingParams(max_tokens=16))  # warmup
    seqs = []
    for p in prompts["gsm8k"][:N_REF]:
        s = eng.add_request(p, sp)
        while eng.has_work():
            eng.step()
        seqs.append(s)
    S["base_rate"] = {1: decode_rate(seqs)}
    print(f"  non-spec bs=1 gsm8k decode {S['base_rate'][1]:.1f} tok/s")
    eng.runner.record_gaps = True
    base_out = {}
    for d in SPEC_DATASETS:
        out, _ = closed_loop(eng, prompts[d], 16, sp)
        base_out[d] = [(s.completion_token_ids, s.logit_gaps) for s in out]
    eng.runner.record_gaps = False
    S["base_outputs"] = base_out
    for c in (4, 16):
        out, wall = closed_loop(eng, prompts["gsm8k"], c, sp)
        S["base_rate"][c] = sum(s.num_completion_tokens for s in out) / wall
        print(f"  non-spec gsm8k c={c}: {S['base_rate'][c]:.1f} output tok/s")
    del eng
    free()


def test_4a_tau_parity_with_reference(report):
    rows, ok = {}, True
    for d in SPEC_DATASETS:
        r, t = S["ref"][d]["tau"], S["ts_tau_bs1"][d]
        rows[d] = {"ref": round(r, 3), "tinyserve": round(t, 3), "ratio": round(t / r, 3)}
        ok &= abs(t / r - 1) <= 0.10
        print(f"M7a {d}: tinyserve tau={t:.3f} reference tau={r:.3f} ratio={t / r:.3f} (within +-10%)")
    report["a_tau_parity"] = rows
    # Diagnostic: how often do tinyserve and the reference produce identical tokens?
    same = sum(a == b for d in SPEC_DATASETS for a, b in zip(S["spec_tokens_bs1"][d], S["ref_outputs"][d]))
    report["a_diag_identical_outputs_vs_reference"] = f"{same}/{N_REF * len(SPEC_DATASETS)}"
    assert ok


def test_4b_tau_vs_paper(report):
    rows, ok = {}, True
    for d in SPEC_DATASETS:
        t, p = S["ts_tau_all"][d], PAPER_TAU_QWEN3_4B[d]
        rows[d] = {"tinyserve": round(t, 3), "paper": p, "ratio": round(t / p, 3), "min": PAPER_MIN[d]}
        ok &= t >= PAPER_MIN[d]
        print(f"M7b {d}: tau={t:.3f} paper={p} ratio={t / p:.3f} (>= {PAPER_MIN[d]})")
    report["b_tau_vs_paper"] = rows
    assert ok


def test_4c_lossless(report):
    verdicts = []
    for d in SPEC_DATASETS:
        for spec, (base, gaps) in zip(S["spec_outputs"][d], S["base_outputs"][d]):
            verdicts.append((d,) + compare_tokens(base, spec, gaps))
    n_exact = sum(v[1] == "exact" for v in verdicts)
    n_ok = sum(v[1] != "mismatch" for v in verdicts)
    report["c_exact"], report["c_exact_or_near_tie"] = f"{n_exact}/96", f"{n_ok}/96"
    report["c_divergences"] = [{"dataset": v[0], "verdict": v[1], "at": v[2], "gap": v[3]} for v in verdicts if v[1] != "exact"]
    print(f"M7c spec vs non-spec: exact={n_exact}/96 (>=90) exact_or_near_tie={n_ok}/96 (96)")
    assert n_exact >= 90 and n_ok == 96


def test_4d_speedup(report):
    sp = {c: S["spec_rate"][c] / S["base_rate"][c] for c in (1, 4, 16)}
    report["d_speed"] = {c: {"dflash_tok_s": round(S["spec_rate"][c], 1), "base_tok_s": round(S["base_rate"][c], 1),
                             "speedup": round(sp[c], 3)} for c in (1, 4, 16)}
    for c in (1, 4, 16):
        print(f"M7d gsm8k c={c}: DFlash {S['spec_rate'][c]:.1f} tok/s vs non-spec (CUDA graphs) {S['base_rate'][c]:.1f} tok/s "
              f"-> {sp[c]:.2f}x" + (" (>= 2.0)" if c == 1 else ""))
    assert sp[1] >= 2.0
