"""flash-attn 2 backend (GPU only), using the nano-vllm call pattern.

  * varlen (prefill, spec verify, DFlash draft): flash_attn_varlen_func.
    With a block_table it reads K/V from the paged cache (needed whenever a
    sequence has context that is not in this forward: a cached prefix, or the
    earlier tokens during verify).
  * decode: flash_attn_with_kvcache, one query per sequence.

FA2's paged KV requires the cache block size to be a multiple of 256.
"""

from __future__ import annotations

import torch

from tinyserve.attention.metadata import AttnMetadata

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:  # CPU machines: the torch backend is used instead
    flash_attn_varlen_func = flash_attn_with_kvcache = None


def attend(q, k, v, k_cache, v_cache, m: AttnMetadata, scale: float) -> torch.Tensor:
    if flash_attn_varlen_func is None:
        raise RuntimeError("flash-attn is not installed; use attn_backend='torch'")
    if m.is_varlen:
        if m.block_tables is not None:
            k, v = k_cache, v_cache
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=m.cu_seqlens_q, cu_seqlens_k=m.cu_seqlens_k,
            max_seqlen_q=m.max_seqlen_q, max_seqlen_k=m.max_seqlen_k,
            softmax_scale=scale, causal=m.causal, block_table=m.block_tables,
        )
    out = flash_attn_with_kvcache(
        q.unsqueeze(1), k_cache, v_cache,
        cache_seqlens=m.context_lens, block_table=m.block_tables,
        softmax_scale=scale, causal=True,
    )
    return out.squeeze(1)
