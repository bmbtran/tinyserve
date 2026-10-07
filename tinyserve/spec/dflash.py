"""DFlash draft model (Chen, Liang, Liu 2026, arXiv:2602.06036) on the paged KV cache.

How the draft works, in one picture. For a sequence whose last token t sits
at position n-1, one draft forward predicts the next 15 tokens AT ONCE:

    keys/values:  [ context: target features of positions 0..n-2 | noise block: t, M, M, ..., M ]
    queries:                                                       [ t, M, M, ..., M ]   (16, non-causal)
    output:       draft guesses d_1..d_15 from the 15 MASK positions

* "Context" = the target's hidden states from 5 of its layers (aux hidden),
  projected by `fc` to one H-wide vector per position and normed. They only
  provide keys/values (they are injected into the draft's KV cache), never
  queries, and they skip `input_layernorm`. They are written to the draft KV
  cache once and reused, exactly like ordinary KV.
* The "noise" block is [t, MASK x 15] embedded with the TARGET's embedding.
  All 16 queries attend to all context and to each other (causal=False):
  this is the block-diffusion part, one parallel denoising step.
* Logits use the TARGET's lm_head. The draft owns no embedding or lm_head.

The draft's KV pool has its own 5 layers but uses the SAME block tables and
slot mapping as the target, so paging, preemption and prefix caching work
unchanged. Noise-block KV at [n-1, n+14] is scratch: the next step
overwrites it (the reference's `past_key_values_draft.crop(start)`).
"""

from __future__ import annotations

import torch
from torch import nn

from tinyserve.attention import Attention
from tinyserve.models.qwen3 import Qwen3MLP, RMSNorm, RotaryEmbedding


def draft_target_layer_ids(draft_cfg) -> list[int]:
    ids = (getattr(draft_cfg, "dflash_config", None) or {}).get("target_layer_ids")
    if ids is not None:
        return list(ids)
    # Same default as the reference build_target_layer_ids().
    n_target, n_draft = draft_cfg.num_target_layers, draft_cfg.num_hidden_layers
    if n_draft == 1:
        return [n_target // 2]
    start, end = 1, n_target - 3
    return [int(round(start + i * (end - start) / (n_draft - 1))) for i in range(n_draft)]


class DFlashAttention(nn.Module):
    def __init__(self, cfg, rope: RotaryEmbedding, backend: str, use_triton_store: bool):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        bias = getattr(cfg, "attention_bias", False)
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=bias)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.rope = [rope]
        self.attn = Attention(self.num_heads, self.num_kv_heads, self.head_dim, backend, use_triton_store)

    def forward(self, x, positions, ctx, ctx_positions):
        """x: normed noise hidden [Nn, H]; ctx: context features [Nc, H] (no layernorm)."""
        n, nc = x.shape[0], ctx.shape[0]
        q = self.q_norm(self.q_proj(x).view(n, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(n, self.num_kv_heads, self.head_dim))
        v = self.v_proj(x).view(n, self.num_kv_heads, self.head_dim)
        k_ctx = self.k_norm(self.k_proj(ctx).view(nc, self.num_kv_heads, self.head_dim))
        v_ctx = self.v_proj(ctx).view(nc, self.num_kv_heads, self.head_dim)
        q, k = self.rope[0](positions, q, k)
        (k_ctx,) = self.rope[0](ctx_positions, k_ctx)
        # Context rows first, then noise rows: the metadata's slot_mapping uses
        # the same order. Only noise rows produce queries.
        o = self.attn(q, torch.cat([k_ctx, k]), torch.cat([v_ctx, v]))
        return self.o_proj(o.reshape(n, -1))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, cfg, rope, backend, use_triton_store):
        super().__init__()
        self.self_attn = DFlashAttention(cfg, rope, backend, use_triton_store)
        self.mlp = Qwen3MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, positions, ctx, ctx_positions):
        x = x + self.self_attn(self.input_layernorm(x), positions, ctx, ctx_positions)
        return x + self.mlp(self.post_attention_layernorm(x))


class DFlashDraftModel(nn.Module):
    """Parameter names match the z-lab checkpoint: fc, hidden_norm, norm, layers.*"""

    def __init__(self, cfg, target, backend: str = "torch", use_triton_store: bool = False,
                 max_positions: int = 8192, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = cfg
        self.target_layer_ids = draft_target_layer_ids(cfg)
        self.block_size = cfg.block_size
        self.mask_token_id = cfg.dflash_config["mask_token_id"]
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.rope = RotaryEmbedding(head_dim, getattr(cfg, "rope_theta", 1e6), max_positions, dtype)
        self.layers = nn.ModuleList(DFlashDecoderLayer(cfg, self.rope, backend, use_triton_store) for _ in range(cfg.num_hidden_layers))
        self.fc = nn.Linear(len(self.target_layer_ids) * cfg.hidden_size, cfg.hidden_size, bias=False)
        self.hidden_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.target = [target]  # shares embed_tokens / lm_head; a list so it is not a submodule

    def attention_layers(self) -> list[Attention]:
        return [layer.self_attn.attn for layer in self.layers]

    def forward(self, ctx_hidden, ctx_positions, noise_ids, noise_positions) -> torch.Tensor:
        """ctx_hidden: target aux hidden [Nc, len(ids)*H]; noise_ids: [Nn]. Returns normed hidden [Nn, H]."""
        ctx = self.hidden_norm(self.fc(ctx_hidden))
        x = self.target[0].embed_tokens(noise_ids)
        for layer in self.layers:
            x = layer(x, noise_positions, ctx, ctx_positions)
        return self.norm(x)

