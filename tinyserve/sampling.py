"""Sampling parameters and the sampler.

Greedy (temperature 0) is a plain argmax, which is what speculative decoding
and every correctness test rely on. Temperature / top-k / top-p are applied
row by row: simple, and fast enough because sampling is a tiny part of a step.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class SamplingParams:
    max_tokens: int = 256
    temperature: float = 0.0  # 0 -> greedy
    top_p: float = 1.0
    top_k: int = -1  # -1 -> disabled
    ignore_eos: bool = False  # needed for fixed-length benchmarks (vLLM has the same flag)
    stop_token_ids: list[int] = field(default_factory=list)
    seed: int | None = None  # per-request seed (OpenAI `seed`); None -> engine generator

    def __post_init__(self):
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.temperature < 0.0:
            raise ValueError("temperature must be >= 0")

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


def filter_top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Mask one row of logits [V] to its top-k tokens, then to the smallest set
    whose probability mass reaches top_p. Masked entries become -inf."""
    logits = logits.clone()
    if 0 < top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits[logits < kth] = float("-inf")
    if top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        # Keep a token if the mass of everything ranked above it is still below
        # top_p. The top-1 token has zero mass before it, so it is always kept.
        mass_before = torch.cumsum(probs, dim=-1) - probs
        logits[order[mass_before >= top_p]] = float("-inf")
    return logits


def sample(
    logits: torch.Tensor,
    params: list[SamplingParams],
    generators: list[torch.Generator | None] | None = None,
) -> list[int]:
    """logits: [B, V]. Returns one token id per row."""
    out = torch.argmax(logits, dim=-1).tolist()
    for i, p in enumerate(params):
        if p.greedy:
            continue
        row = filter_top_k_top_p(logits[i].float() / p.temperature, p.top_k, p.top_p)
        probs = torch.softmax(row, dim=-1)
        gen = generators[i] if generators else None
        out[i] = int(torch.multinomial(probs.cpu(), 1, generator=gen).item())
    return out
