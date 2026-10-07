"""M2 on an L4: flash vs torch backend, Triton store, bf16 greedy vs HF."""

import pytest
import torch

from tests.util import compare_tokens, hf_greedy
from tinyserve.attention import AttnMetadata, set_attn_metadata
from tinyserve.config import EngineConfig
from tinyserve.engine import LLMEngine
from tinyserve.sampling import SamplingParams

pytestmark = pytest.mark.gpu
MODEL = "Qwen/Qwen3-0.6B"

PROMPTS = [
    "The capital of France is",
    "Write a Python function that checks whether a number is prime.",
    "Explain the difference between TCP and UDP.",
    "Once upon a time, in a small village by the sea,",
    "List five uses of a paperclip:",
    "Translate to German: The weather is nice today.",
    "What is 17 times 23? Show your work.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
]


def make_engine(backend, **kw):
    base = dict(model=MODEL, dtype="bfloat16", device="cuda", attn_backend=backend, block_size=256,
                num_kv_blocks=48, max_num_seqs=16, max_model_len=2048, max_num_batched_tokens=4096, enforce_eager=True)
    base.update(kw)
    return LLMEngine(EngineConfig(**base))


@pytest.fixture(scope="module")
def flash_engine():
    return make_engine("flash")


def full_prefill_logits(model, seqs: list[list[int]]):
    lens = [len(s) for s in seqs]
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
    set_attn_metadata(AttnMetadata(is_varlen=True, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=max(lens),
                                   max_seqlen_k=max(lens), q_lens=lens, k_lens=lens))
    try:
        with torch.inference_mode():
            ids = torch.tensor(sum(seqs, []), device="cuda")
            pos = torch.cat([torch.arange(n, device="cuda") for n in lens])
            h, _ = model(ids, pos)
            return model.compute_logits(h).float()
    finally:
        set_attn_metadata(None)


def test_a_flash_vs_torch_backend(flash_engine, report):
    tok = flash_engine.tokenizer
    seqs = [tok.encode(p * 6) for p in PROMPTS[:4]]
    lf = full_prefill_logits(flash_engine.runner.model, seqs)
    # Same weights, torch backend: swap the backend module on every layer.
    from tinyserve.attention import backend_torch

    for layer in flash_engine.runner.layers:
        layer.backend = backend_torch
    try:
        lt = full_prefill_logits(flash_engine.runner.model, seqs)
    finally:
        from tinyserve.attention import backend_flash

        for layer in flash_engine.runner.layers:
            layer.backend = backend_flash
    diff = (lf - lt).abs().max().item()
    top1 = (lf.argmax(-1) == lt.argmax(-1)).float().mean().item()
    report["a_max_abs_diff"] = round(diff, 4)
    report["a_top1_agree"] = round(top1, 4)

    # Diagnostic (does not change the threshold): how far is EACH bf16 backend
    # from a float32 run of the same weights? If both are equally far, the
    # flash/torch gap is bf16 noise, not a backend bug.
    import copy

    caches = [(l.k_cache, l.v_cache) for l in flash_engine.runner.layers]
    for l in flash_engine.runner.layers:
        l.k_cache = l.v_cache = torch.tensor([])
    m32 = copy.deepcopy(flash_engine.runner.model).float()
    for (kc, vc), l in zip(caches, flash_engine.runner.layers):
        l.k_cache, l.v_cache = kc, vc
    for l in m32.attention_layers():
        l.backend = backend_torch
    l32 = full_prefill_logits(m32, seqs)
    del m32
    torch.cuda.empty_cache()
    e_flash = (lf - l32).abs().max().item()
    e_torch = (lt - l32).abs().max().item()
    report["a_diag_flash_vs_fp32"] = round(e_flash, 4)
    report["a_diag_torch_bf16_vs_fp32"] = round(e_torch, 4)
    report["a_diag_flash_top1_vs_fp32"] = round((lf.argmax(-1) == l32.argmax(-1)).float().mean().item(), 4)
    report["a_diag_torch_top1_vs_fp32"] = round((lt.argmax(-1) == l32.argmax(-1)).float().mean().item(), 4)
    report["a_diag_mean_abs_diff"] = round((lf - lt).abs().mean().item(), 5)
    print(f"M2a diag: |flash-fp32|max={e_flash:.4f} |torch_bf16-fp32|max={e_torch:.4f} "
          f"mean|flash-torch|={report['a_diag_mean_abs_diff']}")
    print(f"M2a flash-vs-torch max_abs_diff={diff:.4f} (<0.15) top1_agree={top1:.4f} (>=0.99)")
    assert diff < 0.15 and top1 >= 0.99


def test_b_triton_store_equals_torch_store(report):
    from tinyserve.attention.backend_torch import store_kv
    from tinyserve.attention.triton_store import triton_store_kv

    torch.manual_seed(0)
    nb, bs, h, d, n = 6, 256, 8, 128, 300
    k = torch.randn(n, h, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(n, h, d, device="cuda", dtype=torch.bfloat16)
    slots = torch.randperm(nb * bs, device="cuda")[:n]
    slots[::7] = -1
    caches = [torch.zeros(nb, bs, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    store_kv(k, v, caches[0], caches[1], slots)
    triton_store_kv(k, v, caches[2], caches[3], slots)
    exact = torch.equal(caches[0], caches[2]) and torch.equal(caches[1], caches[3])
    report["b_triton_store_exact"] = exact
    print(f"M2b triton_store == torch_store: {exact}")
    assert exact


def test_c_greedy_vs_hf_bf16(flash_engine, report):
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(flash_engine.model_path, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    tok = flash_engine.tokenizer
    prompts = [tok.encode(p) for p in PROMPTS]
    refs = [hf_greedy(hf, p, 128, return_gaps=True) for p in prompts]
    # Diagnostic baseline: HF against ITSELF with a different attention kernel.
    hf.config._attn_implementation = "eager"
    hf.set_attn_implementation("eager") if hasattr(hf, "set_attn_implementation") else None
    eager = [hf_greedy(hf, p, 128) for p in prompts]
    base = [compare_tokens(r[0], e, r[1])[0] for r, e in zip(refs, eager)]
    report["c_diag_hf_sdpa_vs_hf_eager_exact"] = f"{base.count('exact')}/8"
    report["c_diag_hf_sdpa_vs_hf_eager_ok"] = f"{sum(b != 'mismatch' for b in base)}/8"
    print(f"M2c diag: HF-sdpa vs HF-eager (both bf16) exact={base.count('exact')}/8")
    del hf
    torch.cuda.empty_cache()
    outs = flash_engine.generate(prompts, SamplingParams(max_tokens=128, ignore_eos=True))
    verdicts = []
    for (ref, gaps), o in zip(refs, outs):
        v, idx, gap = compare_tokens(ref, o["token_ids"], gaps)
        verdicts.append({"verdict": v, "diverge_at": idx, "gap": gap})
        print(f"  {v} diverge_at={idx} gap={gap}")
    n_exact = sum(v["verdict"] == "exact" for v in verdicts)
    n_ok = sum(v["verdict"] != "mismatch" for v in verdicts)
    report["c_exact"] = f"{n_exact}/8"
    report["c_exact_or_near_tie"] = f"{n_ok}/8"
    report["c_details"] = verdicts
    print(f"M2c greedy vs HF bf16: exact={n_exact}/8 (>=7) exact_or_near_tie={n_ok}/8 (8)")
    assert n_exact >= 7 and n_ok == 8
