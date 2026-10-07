"""Per-forward attention metadata, shared by every layer through a global.

The model runner builds one AttnMetadata per forward pass (what tokens are
in the batch, where their K/V go, which cache blocks each sequence owns) and
installs it with `set_attn_metadata`. Each Attention layer reads it with
`get_attn_metadata`, so model code never has to thread it through calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AttnMetadata:
    is_varlen: bool  # True: PREFILL / SPEC verify / DFlash draft; False: DECODE (1 query per seq)
    causal: bool = True  # the DFlash draft uses causal=False
    cu_seqlens_q: torch.Tensor | None = None  # int32 [B+1]
    cu_seqlens_k: torch.Tensor | None = None  # int32 [B+1], FULL context length incl. cached prefix
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None  # int64 [num_new_kv]; -1 = do not store
    context_lens: torch.Tensor | None = None  # int32 [B] (decode)
    block_tables: torch.Tensor | None = None  # int32 [B, max_blocks]; None -> attend to the fresh K/V
    # Python copies of the lengths so the torch backend needs no GPU->CPU syncs.
    q_lens: list[int] | None = None
    k_lens: list[int] | None = None


_CURRENT: AttnMetadata | None = None


def set_attn_metadata(m: AttnMetadata | None) -> None:
    global _CURRENT
    _CURRENT = m


def get_attn_metadata() -> AttnMetadata:
    if _CURRENT is None:
        raise RuntimeError("no attention metadata set for this forward pass")
    return _CURRENT
