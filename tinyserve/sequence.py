"""Per-request state.

A Sequence is the scheduler's unit of work: token ids (prompt + generated),
its block table into the paged KV cache, timing for latency metrics, and the
extra bookkeeping DFlash speculative decoding needs (see PLAN.md section 4.3).
"""

from __future__ import annotations

import itertools
import time
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tinyserve.sampling import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


_seq_counter = itertools.count()


class Sequence:
    def __init__(self, token_ids: list[int], params: "SamplingParams", seq_id: int | None = None):
        if not token_ids:
            raise ValueError("a prompt needs at least one token")
        self.seq_id = next(_seq_counter) if seq_id is None else seq_id
        self.status = SequenceStatus.WAITING
        self.params = params
        self.token_ids: list[int] = list(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0  # tokens served from the prefix cache at the last prefill
        self.block_table: list[int] = []
        self.finish_reason: str | None = None  # "stop" | "length"
        self.num_preemptions = 0

        # Timing (time.perf_counter). token_times[i] = when completion token i was produced.
        self.arrival_time = time.perf_counter()
        self.first_token_time: float | None = None
        self.finish_time: float | None = None
        self.token_times: list[float] = []

        # Spec-decode state (PLAN.md 4.3).
        self.draft_ctx_len = 0  # positions [0, draft_ctx_len) have draft context KV written
        self.pending_hidden: Any = None  # target aux hidden for [draft_ctx_len, num_tokens-1)
        self.spec_steps = 0
        self.spec_accepted = 0

        # Free-form slot for the server (e.g. a request handle); the engine never reads it.
        self.user_data: Any = None

    # --- derived quantities -------------------------------------------------
    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_completion_tokens(self) -> int:
        return len(self.token_ids) - self.num_prompt_tokens

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def last_token(self) -> int:
        return self.token_ids[-1]

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    def num_blocks(self, block_size: int) -> int:
        """Blocks needed to hold KV for every current token."""
        return (self.num_tokens + block_size - 1) // block_size

    @property
    def tau(self) -> float | None:
        """Mean acceptance length per spec step, counting the bonus token."""
        if self.spec_steps == 0:
            return None
        return (self.spec_accepted + self.spec_steps) / self.spec_steps

    def append_token(self, token_id: int, now: float | None = None) -> None:
        now = time.perf_counter() if now is None else now
        if self.first_token_time is None:
            self.first_token_time = now
        self.token_ids.append(token_id)
        self.token_times.append(now)

    def __repr__(self) -> str:
        return (
            f"Sequence(id={self.seq_id}, status={self.status.name}, tokens={self.num_tokens}, "
            f"prompt={self.num_prompt_tokens}, blocks={self.block_table})"
        )
