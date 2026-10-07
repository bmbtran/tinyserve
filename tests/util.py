"""Shared test helpers: tiny random Qwen3 models, HF reference greedy, near-tie rule."""

from __future__ import annotations

from pathlib import Path

import torch

NEAR_TIE_GAP = 0.5  # the ONLY allowed relaxation (PLAN.md 6.5)


def tiny_qwen3_config(**kw):
    from transformers import Qwen3Config

    base = dict(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=128, vocab_size=512, tie_word_embeddings=True, max_position_embeddings=2048,
        rope_theta=10000.0, rms_norm_eps=1e-6, initializer_range=0.2, eos_token_id=None, bos_token_id=None,
        pad_token_id=None,
    )
    base.update(kw)
    return Qwen3Config(**base)


def make_tiny_model(path: Path, seed: int = 0, **cfg_kw):
    """Random float64 HF Qwen3 saved to `path` (safetensors). Returns the HF model."""
    from transformers import Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = tiny_qwen3_config(**cfg_kw)
    cfg._attn_implementation = "sdpa"  # eager does softmax in fp32
    model = Qwen3ForCausalLM(cfg).to(torch.float64).eval()
    model.save_pretrained(path, safe_serialization=True)
    return model


def tiny_engine(path, proposer=None, **kw):
    from tinyserve.config import EngineConfig
    from tinyserve.engine import LLMEngine

    base = dict(model=str(path), device="cpu", dtype="float64", attn_backend="torch", block_size=16,
                max_model_len=512, max_num_batched_tokens=2048, max_num_seqs=8)
    base.update(kw)
    return LLMEngine(EngineConfig(**base), proposer=proposer, load_tokenizer=False)


@torch.no_grad()
def hf_greedy(model, prompt: list[int], n: int, return_gaps: bool = False):
    """Greedy continuation from an HF model (with its KV cache)."""
    dev = next(model.parameters()).device
    ids = torch.tensor([prompt], device=dev)
    out = model(ids, use_cache=True)
    past, toks, gaps = out.past_key_values, [], []
    logits = out.logits[0, -1]
    for _ in range(n):
        top2 = logits.float().topk(2).values
        gaps.append((top2[0] - top2[1]).item())
        tok = int(logits.argmax())
        toks.append(tok)
        out = model(torch.tensor([[tok]], device=dev), past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[0, -1]
    return (toks, gaps) if return_gaps else toks


def compare_tokens(ref: list[int], got: list[int], ref_gaps: list[float] | None, thr: float = NEAR_TIE_GAP) -> tuple[str, int | None, float | None]:
    """Near-tie rule: 'exact' if identical; 'near_tie' if they first diverge at
    a step where the reference's top-1/top-2 logit gap < thr; else 'mismatch'.
    Returns (verdict, divergence index, gap there)."""
    n = min(len(ref), len(got))
    for i in range(n):
        if ref[i] != got[i]:
            gap = ref_gaps[i] if ref_gaps is not None and i < len(ref_gaps) else None
            return ("near_tie" if gap is not None and gap < thr else "mismatch"), i, gap
    if len(ref) != len(got):
        gap = ref_gaps[n] if ref_gaps is not None and n < len(ref_gaps) else None
        return ("near_tie" if gap is not None and gap < thr else "mismatch"), n, gap
    return "exact", None, None
