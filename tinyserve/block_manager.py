"""Paged KV-cache block manager with hash-based prefix caching (pure Python).

The KV cache is a pool of fixed-size *blocks*; each holds K/V for
`block_size` consecutive token positions of one sequence. A sequence's
`block_table` lists its blocks in order, so token position `p` lives in slot

    block_table[p // block_size] * block_size + p % block_size

This is the PagedAttention idea (vLLM): no per-sequence contiguous
allocation, so no fragmentation, and blocks can be *shared*.

Prefix caching (like vLLM's automatic prefix caching):
  * A FULL block whose KV is known valid gets a hash that chains the parent
    block's hash with the block's token ids:  h_i = xxh64((h_{i-1}, tokens_i)).
    Chaining means identical tokens at different depths never collide, and a
    hit on block i implies the whole prefix [0, i] matched.
  * A new prompt walks its full blocks, reusing any whose hash is registered
    (ref_count += 1). The first miss ends the match.
  * Hashes are registered only by `commit_full_blocks`, i.e. after the model
    has actually written the KV (and, with DFlash, the draft KV too).
  * Freed blocks keep their hash and stay reusable until the free list hands
    them out again. The free list is FIFO, so that is roughly LRU eviction.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import xxhash

if TYPE_CHECKING:
    from tinyserve.sequence import Sequence


def chain_hash(parent: int | None, token_ids: tuple[int, ...]) -> int:
    h = xxhash.xxh64()
    h.update(b"root" if parent is None else parent.to_bytes(8, "little"))
    h.update(b"".join(t.to_bytes(4, "little", signed=False) for t in token_ids))
    return h.intdigest()


class Block:
    __slots__ = ("block_id", "ref_count", "hash", "token_ids")

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash: int | None = None
        self.token_ids: tuple[int, ...] = ()

    def __repr__(self) -> str:
        return f"Block({self.block_id}, ref={self.ref_count}, hash={self.hash is not None})"


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_cache: bool):
        if num_blocks < 1:
            raise ValueError("need at least one KV block")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_cache = enable_prefix_cache
        self.blocks = [Block(i) for i in range(num_blocks)]
        # Ordered set of free block ids; oldest-freed first (popped first).
        self.free: OrderedDict[int, None] = OrderedDict((i, None) for i in range(num_blocks))
        self.hash_to_block: dict[int, int] = {}
        self.stats = {"prefix_query_tokens": 0, "prefix_hit_tokens": 0}

    # --- helpers -----------------------------------------------------------
    @property
    def num_free_blocks(self) -> int:
        return len(self.free)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - len(self.free)

    @property
    def prefix_hit_rate(self) -> float:
        q = self.stats["prefix_query_tokens"]
        return self.stats["prefix_hit_tokens"] / q if q else 0.0

    def _take_free(self, block_id: int | None = None) -> Block:
        """Remove a block from the free list. A specific id revives a cached
        block; otherwise the oldest free block is recycled and its stale hash
        is forgotten (this is the eviction)."""
        if block_id is None:
            block_id, _ = self.free.popitem(last=False)
            block = self.blocks[block_id]
            if block.hash is not None and self.hash_to_block.get(block.hash) == block_id:
                del self.hash_to_block[block.hash]
            block.hash = None
            block.token_ids = ()
        else:
            del self.free[block_id]
            block = self.blocks[block_id]
        assert block.ref_count == 0
        block.ref_count = 1
        return block

    def _match_prefix(self, seq: "Sequence") -> list[int]:
        """Block ids of the longest registered prefix of seq's full blocks,
        capped so at least one token is left to compute (we need its logits)."""
        if not self.enable_prefix_cache:
            return []
        bs = self.block_size
        hits: list[int] = []
        parent: int | None = None
        for i in range(seq.num_tokens // bs):
            tokens = tuple(seq.token_ids[i * bs : (i + 1) * bs])
            h = chain_hash(parent, tokens)
            block_id = self.hash_to_block.get(h)
            if block_id is None or self.blocks[block_id].token_ids != tokens:
                break
            hits.append(block_id)
            parent = h
        # INVARIANT: num_cached_tokens <= num_tokens - 1. If every token is
        # cached, drop the last hit block and recompute it.
        if hits and len(hits) * bs >= seq.num_tokens:
            hits.pop()
        return hits

    # --- allocation API (prefill) -------------------------------------------
    def can_allocate(self, seq: "Sequence") -> bool:
        hits = self._match_prefix(seq)
        # Hit blocks that are currently free also come out of the free pool.
        revived = sum(1 for b in hits if self.blocks[b].ref_count == 0)
        fresh = seq.num_blocks(self.block_size) - len(hits)
        return fresh + revived <= self.num_free_blocks

    def num_uncached_tokens(self, seq: "Sequence") -> int:
        """Tokens a prefill of seq would have to compute (for the token budget)."""
        return seq.num_tokens - len(self._match_prefix(seq)) * self.block_size

    def allocate(self, seq: "Sequence") -> None:
        assert not seq.block_table, "sequence already holds blocks"
        hits = self._match_prefix(seq)
        for block_id in hits:
            block = self.blocks[block_id]
            if block.ref_count == 0:
                self._take_free(block_id)
            else:
                block.ref_count += 1
            seq.block_table.append(block_id)
        seq.num_cached_tokens = len(hits) * self.block_size
        for _ in range(seq.num_blocks(self.block_size) - len(hits)):
            seq.block_table.append(self._take_free().block_id)
        self.stats["prefix_query_tokens"] += seq.num_tokens
        self.stats["prefix_hit_tokens"] += seq.num_cached_tokens

    def deallocate(self, seq: "Sequence") -> None:
        # Release in reverse so a sequence's tail blocks (least likely to be
        # shared) are recycled before its prefix blocks.
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            assert block.ref_count >= 0
            if block.ref_count == 0:
                self.free[block_id] = None  # hash kept -> still reusable until recycled
        seq.block_table = []
        seq.num_cached_tokens = 0

    # --- decode / spec API --------------------------------------------------
    def _blocks_needed(self, seq: "Sequence", lookahead: int) -> int:
        """Blocks to cover positions [0, num_tokens - 1 + lookahead]."""
        n = seq.num_tokens + lookahead
        return (n + self.block_size - 1) // self.block_size

    def can_append(self, seq: "Sequence", lookahead: int = 0) -> bool:
        return self._blocks_needed(seq, lookahead) - len(seq.block_table) <= self.num_free_blocks

    def ensure_slots(self, seq: "Sequence", lookahead: int = 0) -> None:
        """Grow the block table so it covers positions [0, num_tokens-1+lookahead]
        (decode: lookahead 0 -> a slot for the last token; spec: 15)."""
        need = self._blocks_needed(seq, lookahead) - len(seq.block_table)
        if need > self.num_free_blocks:
            raise RuntimeError("out of KV blocks; call can_append first")
        for _ in range(need):
            seq.block_table.append(self._take_free().block_id)

    def commit_full_blocks(self, seq: "Sequence", num_valid_kv_tokens: int) -> None:
        """Register a hash for every block fully covered by valid KV
        (positions [0, num_valid_kv_tokens)), so later prompts can reuse it."""
        if not self.enable_prefix_cache:
            return
        bs = self.block_size
        parent: int | None = None
        for i in range(min(num_valid_kv_tokens // bs, len(seq.block_table))):
            block = self.blocks[seq.block_table[i]]
            if block.hash is None:
                tokens = tuple(seq.token_ids[i * bs : (i + 1) * bs])
                block.hash = chain_hash(parent, tokens)
                block.token_ids = tokens
                # If another block already serves this hash (two sequences
                # computed the same prefix concurrently), keep the first.
                self.hash_to_block.setdefault(block.hash, block.block_id)
            parent = block.hash

    # --- debugging -----------------------------------------------------------
    def check_invariants(self, live_seqs: list["Sequence"]) -> None:
        """Assert the pool is consistent with the given live sequences."""
        expected = [0] * self.num_blocks
        for s in live_seqs:
            for b in s.block_table:
                expected[b] += 1
        for b in self.blocks:
            assert b.ref_count == expected[b.block_id], (b, expected[b.block_id])
            assert (b.ref_count == 0) == (b.block_id in self.free), b
        assert len(self.free) + sum(1 for b in self.blocks if b.ref_count > 0) == self.num_blocks
        for h, bid in self.hash_to_block.items():
            assert self.blocks[bid].hash == h
