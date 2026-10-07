"""Draft proposers for speculative decoding.

A Proposer returns K = spec_block_size - 1 draft tokens per sequence. The
target verifies them (model_runner.run_spec), so correctness never depends
on the proposer, only the acceptance length does. Mock proposers make that
testable independently of draft quality:
  OracleProposer: the target's own greedy continuation (precomputed) -> tau = 16
  RandomProposer: random tokens -> tau ~ 1
"""

from __future__ import annotations

from typing import Protocol

import torch

from tinyserve.attention import AttnMetadata, set_attn_metadata
from tinyserve.sequence import Sequence


class Proposer(Protocol):
    def propose(self, seqs: list[Sequence], runner) -> torch.Tensor:
        """Return [B, spec_block_size - 1] draft token ids."""
        ...


class OracleProposer:
    def __init__(self, continuations: dict[tuple[int, ...], list[int]], k: int = 15):
        self.continuations = continuations  # prompt tuple -> greedy completion (long enough)
        self.k = k

    def propose(self, seqs, runner) -> torch.Tensor:
        rows = []
        for s in seqs:
            cont = self.continuations[tuple(s.token_ids[: s.num_prompt_tokens])]
            c = s.num_completion_tokens
            row = cont[c : c + self.k]
            rows.append(row + [0] * (self.k - len(row)))
        return torch.tensor(rows, dtype=torch.long)


class RandomProposer:
    def __init__(self, vocab_size: int, k: int = 15, seed: int = 0):
        self.vocab_size, self.k = vocab_size, k
        self.gen = torch.Generator().manual_seed(seed)

    def propose(self, seqs, runner) -> torch.Tensor:
        return torch.randint(0, self.vocab_size, (len(seqs), self.k), generator=self.gen)


class DFlashProposer:
    """One batched, non-causal draft forward (PLAN.md 4.3 step 2)."""

    def __init__(self, draft_model):
        self.draft = draft_model
        self.record_logits = False  # tests: keep each step's draft logits per sequence
        self.logits_log: dict[int, list[torch.Tensor]] = {}

    @torch.inference_mode()
    def propose(self, seqs: list[Sequence], runner) -> torch.Tensor:
        blk = runner.cfg.spec_block_size
        mask = self.draft.mask_token_id
        ctx_hidden, ctx_pos, ctx_slots = [], [], []
        noise_ids, noise_pos, noise_slots, k_lens = [], [], [], []
        for s in seqs:
            n, start = s.num_tokens, s.draft_ctx_len
            assert s.pending_hidden is not None and s.pending_hidden.shape[0] == n - 1 - start, \
                (s.seq_id, None if s.pending_hidden is None else s.pending_hidden.shape, n, start)
            ctx_hidden.append(s.pending_hidden)
            ctx_pos += range(start, n - 1)
            ctx_slots += runner._slots(s, start, n - 1)
            noise_ids += [s.last_token] + [mask] * (blk - 1)
            noise_pos += range(n - 1, n - 1 + blk)
            noise_slots += runner._slots(s, n - 1, n - 1 + blk)
            k_lens.append(n - 1 + blk)
        t = runner._t
        meta = AttnMetadata(
            is_varlen=True, causal=False,
            cu_seqlens_q=t([0, *[blk * (i + 1) for i in range(len(seqs))]], torch.int32),
            cu_seqlens_k=t([0, *_cumsum(k_lens)], torch.int32),
            max_seqlen_q=blk, max_seqlen_k=max(k_lens),
            slot_mapping=t(ctx_slots + noise_slots), block_tables=runner._block_tables(seqs),
            q_lens=[blk] * len(seqs), k_lens=k_lens,
        )
        set_attn_metadata(meta)
        x = self.draft(torch.cat(ctx_hidden), t(ctx_pos), t(noise_ids), t(noise_pos))
        h = x.view(len(seqs), blk, -1)[:, 1:]  # the 15 MASK positions predict d_1..d_15
        logits = self.draft.target[0].compute_logits(h)
        if self.record_logits:
            for s, lg in zip(seqs, logits):
                self.logits_log.setdefault(s.seq_id, []).append(lg.detach().clone())
        for s in seqs:
            s.draft_ctx_len = s.num_tokens - 1  # context KV now covers [0, n-2]
            s.pending_hidden = None
        return logits.argmax(-1)


def _cumsum(xs):
    out, total = [], 0
    for x in xs:
        total += x
        out.append(total)
    return out


def make_proposer(cfg, runner) -> Proposer:
    k = cfg.spec_block_size - 1
    if cfg.spec_method == "dflash":
        return DFlashProposer(runner.draft_model)
    if cfg.spec_method == "random":
        return RandomProposer(runner.hf_config.vocab_size, k, cfg.seed)
    raise ValueError(f"spec_method {cfg.spec_method!r} needs an explicit proposer (tests)")
