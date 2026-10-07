"""Reference paged attention in plain PyTorch.

Slow (a Python loop over sequences) but obviously correct, works on CPU with
any block size and any dtype (float64 for exact tests). It is the oracle the
flash-attn backend is checked against.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tinyserve.attention.metadata import AttnMetadata


def store_kv(k: torch.Tensor, v: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor) -> None:
    """Write k/v rows [N, Hkv, D] into the flat cache slots; slot -1 = skip.

    The -1 check needs a GPU->CPU sync, which is illegal while capturing a
    CUDA graph; graph replays therefore never use -1 (padded rows point at a
    reserved dummy block instead)."""
    flat_k = k_cache.view(-1, *k_cache.shape[2:])
    flat_v = v_cache.view(-1, *v_cache.shape[2:])
    if not (k.is_cuda and torch.cuda.is_current_stream_capturing()):
        keep = slot_mapping >= 0
        if not bool(keep.all()):
            k, v, slot_mapping = k[keep], v[keep], slot_mapping[keep]
    flat_k.index_copy_(0, slot_mapping, k)
    flat_v.index_copy_(0, slot_mapping, v)


def _gather(cache: torch.Tensor, block_table: torch.Tensor, k_len: int) -> torch.Tensor:
    """K or V for positions [0, k_len) of one sequence: [k_len, Hkv, D]."""
    bs = cache.shape[1]
    pos = torch.arange(k_len, device=cache.device)
    slots = block_table[pos // bs].long() * bs + pos % bs
    return cache.view(-1, *cache.shape[2:])[slots]


def _attend_one(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, causal: bool) -> torch.Tensor:
    """q: [Lq, Hq, D]; k, v: [Lk, Hkv, D]. Query i sits at absolute position
    Lk - Lq + i and (if causal) may see keys j <= that position."""
    lq, hq, _ = q.shape
    lk, hkv, _ = k.shape
    if hq != hkv:  # GQA: each KV head serves hq/hkv query heads
        k = k.repeat_interleave(hq // hkv, dim=1)
        v = v.repeat_interleave(hq // hkv, dim=1)
    mask = None
    if causal:
        qpos = torch.arange(lk - lq, lk, device=q.device)
        mask = torch.arange(lk, device=q.device)[None, :] <= qpos[:, None]  # True = may attend
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), attn_mask=mask, scale=scale
    )
    return out.transpose(0, 1)


def attend(q, k, v, k_cache, v_cache, m: AttnMetadata, scale: float) -> torch.Tensor:
    if not m.is_varlen:
        # DECODE: one query per sequence; it sees its whole context.
        k_lens = m.k_lens if m.k_lens is not None else m.context_lens.tolist()
        outs = [
            _attend_one(q[b : b + 1], _gather(k_cache, m.block_tables[b], kl), _gather(v_cache, m.block_tables[b], kl), scale, False)
            for b, kl in enumerate(k_lens)
        ]
        return torch.cat(outs, dim=0)

    q_lens = m.q_lens if m.q_lens is not None else (m.cu_seqlens_q[1:] - m.cu_seqlens_q[:-1]).tolist()
    k_lens = m.k_lens if m.k_lens is not None else (m.cu_seqlens_k[1:] - m.cu_seqlens_k[:-1]).tolist()
    outs, q0, k0 = [], 0, 0
    for b, (ql, kl) in enumerate(zip(q_lens, k_lens)):
        qb = q[q0 : q0 + ql]
        if m.block_tables is None:  # plain prefill: K/V are the fresh tensors
            kb, vb = k[k0 : k0 + kl], v[k0 : k0 + kl]
        else:  # prefix-cached prefill / spec verify / draft: read the paged cache
            kb, vb = _gather(k_cache, m.block_tables[b], kl), _gather(v_cache, m.block_tables[b], kl)
        outs.append(_attend_one(qb, kb, vb, scale, m.causal))
        q0, k0 = q0 + ql, k0 + kl
    return torch.cat(outs, dim=0)
