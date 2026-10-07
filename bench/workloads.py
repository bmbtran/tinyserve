"""Benchmark workloads (PLAN.md 7), regenerated bit-identically from seed 0.

W1 random:        256 requests, input length ~ U[256, 768], output 256 (ignore_eos).
W2 shared-prefix: 8 groups x 16 requests; each group shares a 2048-token
                  prefix + a 64-128 token unique suffix; output 128.
W3 spec:          gsm8k / humaneval / mtbench chat prompts (bench/data), greedy,
                  max_tokens 512, thinking disabled.

Synthetic prompts are random *token ids* below 151643 (no special tokens), so
both engines see exactly the same lengths via /v1/completions token prompts.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
VOCAB_REGULAR = 151643  # Qwen3 ids >= this are special tokens


def _ids(rng: random.Random, n: int) -> list[int]:
    return [rng.randrange(VOCAB_REGULAR) for _ in range(n)]


def w1_random(n: int = 256, lo: int = 256, hi: int = 768, out: int = 256, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    return [{"prompt": _ids(rng, rng.randint(lo, hi)), "max_tokens": out} for _ in range(n)]


def fixed_len(n: int, in_len: int, out: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    return [{"prompt": _ids(rng, in_len), "max_tokens": out} for _ in range(n)]


def w2_shared_prefix(groups: int = 8, per_group: int = 16, prefix_len: int = 2048, suffix: tuple[int, int] = (64, 128),
                     out: int = 128, seed: int = 0) -> list[dict]:
    """Requests are ordered group by group (a group's requests arrive together)."""
    rng = random.Random(seed)
    reqs = []
    for g in range(groups):
        prefix = _ids(rng, prefix_len)
        for _ in range(per_group):
            reqs.append({"prompt": prefix + _ids(rng, rng.randint(*suffix)), "max_tokens": out, "group": g})
    return reqs


def load_chat(name: str, n: int | None = None) -> list[dict]:
    """bench/data/{name}.jsonl -> [{"messages": [...], "source": ..., "id": ...}]"""
    rows = [json.loads(line) for line in (DATA / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:n] if n else rows


SPEC_DATASETS = ("gsm8k", "humaneval", "mtbench")
PAPER_TAU_QWEN3_4B = {"gsm8k": 6.53, "humaneval": 6.64, "mtbench": 4.35}  # DFlash Table 1
