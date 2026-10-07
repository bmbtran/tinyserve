"""Qwen3 dense model (0.6B / 1.7B / 4B / 8B) written for a paged-KV engine.

Inputs are FLAT: all tokens of all sequences in the batch are concatenated
into one [N] vector with an explicit position per token. The attention layer
uses the per-forward AttnMetadata to know where each sequence starts.

Module and parameter names mirror Hugging Face (`model.layers.3.self_attn.q_proj`
...) so safetensors checkpoints load by name. Numerics mirror HF too (RMSNorm
in float32, RoPE tables computed in float32) so float64 CPU tests can demand
near-bitwise agreement with `transformers`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tinyserve.attention import Attention


class RMSNorm(nn.Module):
    """y = w * x / sqrt(mean(x^2) + eps), computed in float32 like HF."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.to(torch.float32)
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Neox-style RoPE with a precomputed cos/sin table indexed by position.

    Rotating query/key pairs by a position-dependent angle makes q.k depend
    only on the *relative* distance between tokens."""

    def __init__(self, head_dim: int, theta: float, max_positions: int, dtype: torch.dtype):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(torch.float) / head_dim))
        t = torch.arange(max_positions, dtype=torch.float32)
        freqs = t[:, None] * inv_freq[None, :]  # float32, like HF
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin", emb.sin().to(dtype), persistent=False)

    def forward(self, positions: torch.Tensor, *xs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """positions: [N]; each x: [N, heads, head_dim]."""
        cos = self.cos[positions].unsqueeze(1)
        sin = self.sin[positions].unsqueeze(1)
        return tuple((x * cos) + (rotate_half(x) * sin) for x in xs)


class Qwen3Attention(nn.Module):
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
        # Qwen3's QK-norm: RMSNorm over each head's vector, BEFORE RoPE.
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.rope = [rope]  # list: share the table without registering it as a submodule
        self.attn = Attention(self.num_heads, self.num_kv_heads, self.head_dim, backend, use_triton_store)

    def forward(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        q = self.q_norm(self.q_proj(x).view(n, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(n, self.num_kv_heads, self.head_dim))
        v = self.v_proj(x).view(n, self.num_kv_heads, self.head_dim)
        q, k = self.rope[0](positions, q, k)
        o = self.attn(q, k, v)
        return self.o_proj(o.reshape(n, -1))


class Qwen3MLP(nn.Module):
    """SwiGLU: down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, cfg, rope, backend, use_triton_store):
        super().__init__()
        self.self_attn = Qwen3Attention(cfg, rope, backend, use_triton_store)
        self.mlp = Qwen3MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(positions, self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3Model(nn.Module):
    def __init__(self, cfg, backend, use_triton_store, max_positions, dtype):
        super().__init__()
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.rope = RotaryEmbedding(head_dim, getattr(cfg, "rope_theta", 1e6), max_positions, dtype)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Qwen3DecoderLayer(cfg, self.rope, backend, use_triton_store) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, hf_config, aux_layer_ids: list[int] | None = None, backend: str = "torch",
                 use_triton_store: bool = False, max_positions: int = 8192, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = hf_config
        self.aux_layer_ids = list(aux_layer_ids or [])
        self.model = Qwen3Model(hf_config, backend, use_triton_store, max_positions, dtype)
        self.lm_head = nn.Linear(hf_config.hidden_size, hf_config.vocab_size, bias=False)
        if getattr(hf_config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    @property
    def embed_tokens(self) -> nn.Embedding:
        return self.model.embed_tokens

    def attention_layers(self) -> list[Attention]:
        return [layer.self_attn.attn for layer in self.model.layers]

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (final normed hidden [N, H], aux hidden [N, len(aux)*H] or None).

        Aux hidden = OUTPUT of decoder layer i for each i in aux_layer_ids (HF's
        output_hidden_states[i + 1]): the residual stream, before the final norm.
        The DFlash draft conditions on these."""
        x = self.model.embed_tokens(input_ids)
        aux = []
        for i, layer in enumerate(self.model.layers):
            x = layer(positions, x)
            if i in self.aux_layer_ids:
                aux.append(x)
        aux_hidden = torch.cat(aux, dim=-1) if aux else None
        return self.model.norm(x), aux_hidden

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden is ALREADY normed."""
        return self.lm_head(hidden)
