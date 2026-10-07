import random

import pytest

from tinyserve.block_manager import BlockManager
from tinyserve.sampling import SamplingParams
from tinyserve.sequence import Sequence

BS = 16


def mkseq(tokens):
    return Sequence(list(tokens), SamplingParams())


def run_prefill(bm, seq):
    """Allocate and pretend the model computed every token (KV valid for all)."""
    bm.allocate(seq)
    bm.commit_full_blocks(seq, seq.num_tokens)


def test_allocate_deallocate_refcounts():
    bm = BlockManager(10, BS, enable_prefix_cache=False)
    s = mkseq(range(40))  # 3 blocks
    assert bm.can_allocate(s)
    bm.allocate(s)
    assert len(s.block_table) == 3
    assert bm.num_free_blocks == 7
    assert all(bm.blocks[b].ref_count == 1 for b in s.block_table)
    bm.check_invariants([s])
    bm.deallocate(s)
    assert bm.num_free_blocks == 10
    assert s.block_table == []
    bm.check_invariants([])


def test_free_list_conservation_multiple():
    bm = BlockManager(20, BS, enable_prefix_cache=False)
    seqs = [mkseq(range(n)) for n in (1, 16, 17, 33)]
    for s in seqs:
        bm.allocate(s)
        assert bm.num_free_blocks + bm.num_used_blocks == 20
    assert bm.num_used_blocks == 1 + 1 + 2 + 3
    bm.check_invariants(seqs)
    for s in seqs:
        bm.deallocate(s)
    assert bm.num_free_blocks == 20


@pytest.mark.parametrize("n_tokens", [1, 15, 16, 17, 31, 32])
def test_ensure_slots_lookahead0(n_tokens):
    bm = BlockManager(10, BS, enable_prefix_cache=False)
    s = mkseq(range(n_tokens))
    bm.allocate(s)
    s.token_ids.append(999)  # a decode step adds one token without KV
    bm.ensure_slots(s, 0)
    # positions [0, num_tokens - 1] covered
    assert len(s.block_table) * BS >= s.num_tokens
    assert (len(s.block_table) - 1) * BS < s.num_tokens
    bm.check_invariants([s])


@pytest.mark.parametrize("n_tokens", [1, 2, 16, 17, 30])
def test_ensure_slots_lookahead15(n_tokens):
    bm = BlockManager(10, BS, enable_prefix_cache=False)
    s = mkseq(range(n_tokens))
    bm.allocate(s)
    bm.ensure_slots(s, 15)
    last_pos = s.num_tokens - 1 + 15
    assert len(s.block_table) == last_pos // BS + 1
    bm.check_invariants([s])


def test_can_allocate_false_when_full():
    bm = BlockManager(4, BS, enable_prefix_cache=False)
    a = mkseq(range(48))  # 3 blocks
    bm.allocate(a)
    assert not bm.can_allocate(mkseq(range(17)))  # needs 2, only 1 free
    assert bm.can_allocate(mkseq(range(16)))
    assert not bm.can_append(a, lookahead=BS + 1)
    with pytest.raises(RuntimeError):
        bm.ensure_slots(a, lookahead=2 * BS)


def test_prefix_same_prompt_twice():
    for L in (5, 16, 17, 32, 33, 100):
        bm = BlockManager(64, BS, enable_prefix_cache=True)
        a = mkseq(range(L))
        run_prefill(bm, a)
        b = mkseq(range(L))
        bm.allocate(b)
        assert b.num_cached_tokens == ((L - 1) // BS) * BS, L
        assert b.block_table[: b.num_cached_tokens // BS] == a.block_table[: b.num_cached_tokens // BS]
        bm.check_invariants([a, b])


def test_prefix_divergent_block():
    bm = BlockManager(64, BS, enable_prefix_cache=True)
    base = list(range(100))
    run_prefill(bm, mkseq(base))
    for k in range(6):
        other = list(base)
        other[k * BS + 3] = 5000  # change a token in block k
        s = mkseq(other)
        bm.allocate(s)
        assert s.num_cached_tokens == k * BS
        bm.deallocate(s)


def test_chained_hash_no_collision_at_different_depth():
    bm = BlockManager(64, BS, enable_prefix_cache=True)
    blk = list(range(BS))
    # Prompt A = [X, blk]; prompt B = [blk, ...]. B's first block has the same
    # tokens as A's second block, but a different parent, so it must not hit.
    a = mkseq([7] * BS + blk + [1])
    run_prefill(bm, a)
    b = mkseq(blk + [2] * 5)
    bm.allocate(b)
    assert b.num_cached_tokens == 0


def test_whole_prompt_cached_invariant():
    bm = BlockManager(64, BS, enable_prefix_cache=True)
    a = mkseq(range(48))
    run_prefill(bm, a)
    b = mkseq(range(48))  # exactly 3 full blocks, all cached
    bm.allocate(b)
    assert b.num_cached_tokens == 32  # last block recomputed
    assert b.num_cached_tokens <= b.num_prompt_tokens - 1
    assert len(b.block_table) == 3
    assert b.block_table[2] != a.block_table[2]


def test_freed_block_reusable_until_reallocated():
    bm = BlockManager(4, BS, enable_prefix_cache=True)
    a = mkseq(range(33))  # 3 blocks, 2 full -> hashed
    run_prefill(bm, a)
    bm.deallocate(a)
    assert bm.num_free_blocks == 4
    b = mkseq(range(33))
    assert bm.can_allocate(b)
    bm.allocate(b)
    assert b.num_cached_tokens == 32
    bm.deallocate(b)
    # Now churn through the whole pool with unrelated tokens -> cache evicted.
    c = mkseq([9000 + i for i in range(64)])
    bm.allocate(c)
    bm.deallocate(c)
    d = mkseq(range(33))
    bm.allocate(d)
    assert d.num_cached_tokens == 0
    bm.check_invariants([d])


def test_can_allocate_counts_revived_free_blocks():
    bm = BlockManager(3, BS, enable_prefix_cache=True)
    a = mkseq(range(33))  # 3 blocks
    run_prefill(bm, a)
    bm.deallocate(a)
    b = mkseq(range(33))  # 2 hits (both free) + 1 fresh = 3 from pool
    assert bm.can_allocate(b)
    bm.allocate(b)
    assert bm.num_free_blocks == 0
    bm.check_invariants([b])


def test_commit_respects_valid_kv_count():
    bm = BlockManager(16, BS, enable_prefix_cache=True)
    a = mkseq(range(64))
    bm.allocate(a)
    bm.commit_full_blocks(a, 20)  # only block 0 fully valid
    b = mkseq(range(64))
    bm.allocate(b)
    assert b.num_cached_tokens == 16


def test_stats_hit_rate():
    bm = BlockManager(64, BS, enable_prefix_cache=True)
    run_prefill(bm, mkseq(range(64)))
    s = mkseq(range(64))
    bm.allocate(s)
    assert bm.stats["prefix_query_tokens"] == 128
    assert bm.stats["prefix_hit_tokens"] == 48
    assert bm.prefix_hit_rate == pytest.approx(48 / 128)


def test_randomized_stress():
    rng = random.Random(0)
    bm = BlockManager(40, BS, enable_prefix_cache=True)
    live: list[Sequence] = []
    prefixes = [[rng.randrange(100) for _ in range(rng.randrange(1, 70))] for _ in range(5)]
    for _ in range(1000):
        op = rng.random()
        if op < 0.35:
            toks = list(rng.choice(prefixes)) + [rng.randrange(100) for _ in range(rng.randrange(0, 20))]
            s = mkseq(toks)
            if bm.can_allocate(s):
                bm.allocate(s)
                assert s.num_cached_tokens <= s.num_tokens - 1
                assert s.num_cached_tokens % BS == 0
                bm.commit_full_blocks(s, s.num_tokens)
                live.append(s)
        elif op < 0.75 and live:
            s = rng.choice(live)
            look = rng.choice([0, 15])
            s.token_ids.append(rng.randrange(100))
            if bm.can_append(s, look):
                bm.ensure_slots(s, look)
                assert len(s.block_table) * BS >= s.num_tokens + look
                bm.commit_full_blocks(s, s.num_tokens - 1)
            else:
                s.token_ids.pop()
        elif live:
            s = live.pop(rng.randrange(len(live)))
            bm.deallocate(s)
        bm.check_invariants(live)
        assert bm.num_free_blocks + bm.num_used_blocks == bm.num_blocks
    for s in live:
        bm.deallocate(s)
    bm.check_invariants([])
    assert bm.num_free_blocks == 40
