"""Attention layer + backend selection.

`Attention` is the only place that touches the KV cache: it (1) stores this
forward's new K/V at `slot_mapping`, then (2) attends using the backend.
The model runner assigns `k_cache` / `v_cache` after allocating the pool.
"""

from __future__ import annotations

import torch
from torch import nn

from tinyserve.attention import backend_flash, backend_torch
from tinyserve.attention.metadata import AttnMetadata, get_attn_metadata, set_attn_metadata

__all__ = ["Attention", "AttnMetadata", "get_attn_metadata", "set_attn_metadata", "get_backend"]


def get_backend(name: str):
    return {"flash": backend_flash, "torch": backend_torch}[name]


class Attention(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int, backend: str = "torch", use_triton_store: bool = False):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, head_dim
        self.scale = head_dim**-0.5
        self.backend = get_backend(backend)
        self.use_triton_store = use_triton_store
        # Empty until the model runner allocates the KV pool.
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """q: [Nq, Hq, D]; k, v: [Nk, Hkv, D] (Nk may exceed Nq for the DFlash
        draft, whose context tokens contribute keys but no queries)."""
        m = get_attn_metadata()
        if self.k_cache.numel() and m.slot_mapping is not None:
            if self.use_triton_store:
                from tinyserve.attention.triton_store import triton_store_kv

                triton_store_kv(k, v, self.k_cache, self.v_cache, m.slot_mapping)
            else:
                backend_torch.store_kv(k, v, self.k_cache, self.v_cache, m.slot_mapping)
        return self.backend.attend(q, k, v, self.k_cache, self.v_cache, m, self.scale)
