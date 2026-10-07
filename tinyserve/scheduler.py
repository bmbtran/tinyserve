"""Iteration-level scheduler (continuous batching, Orca-style). Pure Python.

Every engine step asks `schedule()` for ONE batch:
  1. PREFILL first: admit waiting sequences while the batch fits in
     max_num_seqs, the token budget (uncached prompt tokens only) and the
     free KV blocks. New requests therefore join the running set at the next
     step instead of waiting for the whole batch to finish.
  2. Otherwise DECODE (or SPEC when speculative decoding is on) over every
     running sequence. If the KV pool cannot give a sequence its next slots,
     the most recently admitted sequence is *preempted by recompute*: its
     blocks are freed and it goes back to the FRONT of the waiting queue with
     prompt + generated-so-far as its new prompt. Its prefix-cached blocks
     often survive, so the recompute is usually cheap.

`postprocess()` appends the model's new tokens, applies stop conditions and
releases finished sequences' blocks.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

from tinyserve.block_manager import BlockManager
from tinyserve.config import EngineConfig
from tinyserve.sequence import Sequence, SequenceStatus


class BatchKind(Enum):
    PREFILL = auto()
    DECODE = auto()
    SPEC = auto()


@dataclass
class Batch:
    kind: BatchKind
    seqs: list[Sequence]


class Scheduler:
    def __init__(self, cfg: EngineConfig, block_manager: BlockManager, eos_token_id: int | list[int] | None):
        self.cfg = cfg
        self.bm = block_manager
        if eos_token_id is None:
            eos_token_id = []
        self.eos_ids = set([eos_token_id] if isinstance(eos_token_id, int) else eos_token_id)
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_preemptions = 0
        self.decode_kind = BatchKind.SPEC if cfg.spec_method else BatchKind.DECODE
        # With DFlash, a block may only be cached once its DRAFT KV is valid too.
        self.track_draft_kv = cfg.spec_method == "dflash"

    def add(self, seq: Sequence) -> None:
        if seq.num_tokens >= self.cfg.max_model_len:
            raise ValueError(f"prompt has {seq.num_tokens} tokens; max_model_len is {self.cfg.max_model_len}")
        if self.cfg.spec_method and not seq.params.greedy:
            raise ValueError("speculative decoding is greedy-only (temperature must be 0)")
        # A sequence that could not fit in the whole pool even alone would be
        # preempted forever; reject it up front.
        max_len = min(self.cfg.max_model_len, seq.num_tokens + seq.params.max_tokens)
        bs = self.bm.block_size
        if (max_len + self.cfg.spec_lookahead + bs - 1) // bs > self.bm.num_blocks:
            raise ValueError("request can never fit in the KV cache; raise num_kv_blocks")
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    # --- scheduling ----------------------------------------------------------
    def schedule(self) -> Batch | None:
        prefill = self._schedule_prefill()
        if prefill:
            return Batch(BatchKind.PREFILL, prefill)
        if not self.running:
            return None

        lookahead = self.cfg.spec_lookahead
        scheduled: list[Sequence] = []
        # Iterate over a snapshot; preemption shrinks self.running from the end.
        for seq in list(self.running):
            if seq.status != SequenceStatus.RUNNING:
                continue  # preempted earlier in this loop
            while not self.bm.can_append(seq, lookahead):
                victim = self.running[-1]
                self._preempt(victim)
                if victim is seq:
                    break
            else:
                self.bm.ensure_slots(seq, lookahead)
                scheduled.append(seq)
        if not scheduled:
            return None
        return Batch(self.decode_kind, scheduled)

    def _schedule_prefill(self) -> list[Sequence]:
        batch: list[Sequence] = []
        budget = self.cfg.max_num_batched_tokens
        while self.waiting and len(self.running) < self.cfg.max_num_seqs:
            seq = self.waiting[0]
            cost = self.bm.num_uncached_tokens(seq)
            if cost > budget or not self.bm.can_allocate(seq):
                break
            self.waiting.popleft()
            self.bm.allocate(seq)
            budget -= seq.num_tokens - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            batch.append(seq)
        return batch

    def _preempt(self, seq: Sequence) -> None:
        """Free seq's KV and requeue it at the front; it will be re-prefilled
        over prompt + generated tokens (recompute)."""
        self.running.remove(seq)
        self.bm.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        seq.num_preemptions += 1
        seq.draft_ctx_len = 0
        seq.pending_hidden = None
        self.num_preemptions += 1
        self.waiting.appendleft(seq)

    # --- after the model ran ---------------------------------------------------
    def postprocess(self, batch: Batch, new_tokens: list[list[int]]) -> list[Sequence]:
        """Append each sequence's new tokens (1 for prefill/decode, 1..16 for
        spec), stopping early at EOS / stop ids / max_tokens / max_model_len.
        Returns the sequences that finished."""
        finished: list[Sequence] = []
        now = time.perf_counter()
        for seq, toks in zip(batch.seqs, new_tokens, strict=True):
            for tok in toks:
                seq.append_token(tok, now)
                reason = self._stop_reason(seq, tok)
                if reason:
                    seq.finish_reason = reason
                    break
            if seq.finish_reason:
                seq.status = SequenceStatus.FINISHED
                seq.finish_time = now
                self.running.remove(seq)
                self.bm.deallocate(seq)
                seq.pending_hidden = None
                finished.append(seq)
                continue
            # Target KV is valid for positions [0, num_tokens - 2].
            valid = seq.num_tokens - 1
            if self.track_draft_kv:
                valid = min(valid, seq.draft_ctx_len)
            self.bm.commit_full_blocks(seq, valid)
        return finished

    def _stop_reason(self, seq: Sequence, tok: int) -> str | None:
        p = seq.params
        if not p.ignore_eos and tok in self.eos_ids:
            return "stop"
        if tok in p.stop_token_ids:
            return "stop"
        if seq.num_completion_tokens >= p.max_tokens:
            return "length"
        if seq.num_tokens >= self.cfg.max_model_len:
            return "length"
        return None

    def abort(self, seq: Sequence) -> None:
        """Drop a request (e.g. client disconnected)."""
        if seq in self.running:
            self.running.remove(seq)
            self.bm.deallocate(seq)
        elif seq in self.waiting:
            self.waiting.remove(seq)
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = seq.finish_reason or "abort"
        seq.pending_hidden = None
