"""Scheduler tests with a fake 'model' that emits deterministic tokens."""

import pytest

from tinyserve.block_manager import BlockManager
from tinyserve.config import EngineConfig
from tinyserve.sampling import SamplingParams
from tinyserve.scheduler import BatchKind, Scheduler
from tinyserve.sequence import Sequence, SequenceStatus

EOS = 0
BS = 16


def mkcfg(**kw):
    base = dict(device="cpu", attn_backend="torch", block_size=BS, max_model_len=512, max_num_batched_tokens=512)
    base.update(kw)
    return EngineConfig(**base).validate()


def mksched(num_blocks=64, **kw):
    cfg = mkcfg(**kw)
    bm = BlockManager(num_blocks, cfg.block_size, cfg.enable_prefix_cache)
    return Scheduler(cfg, bm, EOS), bm


def fake_next(seq):
    """Deterministic 'model': next token depends only on the full history."""
    return 1 + (sum(seq.token_ids) * 31 + seq.num_tokens) % 97


def drive(sched, max_steps=10_000):
    """Run until no work; returns finished seqs in finish order + kinds seen."""
    finished, kinds = [], []
    for _ in range(max_steps):
        if not sched.has_work():
            break
        batch = sched.schedule()
        assert batch is not None
        kinds.append(batch.kind)
        finished += sched.postprocess(batch, [[fake_next(s)] for s in batch.seqs])
        sched.bm.check_invariants(sched.running)
    return finished, kinds


def test_prefill_admission_respects_max_num_seqs():
    sched, _ = mksched(max_num_seqs=3)
    for i in range(5):
        sched.add(Sequence([i + 1] * 10, SamplingParams(max_tokens=4)))
    batch = sched.schedule()
    assert batch.kind == BatchKind.PREFILL
    assert len(batch.seqs) == 3
    assert len(sched.waiting) == 2


def test_prefill_admission_respects_token_budget():
    sched, _ = mksched(max_num_batched_tokens=512, max_model_len=512)
    for _ in range(4):
        sched.add(Sequence(list(range(1, 201)), SamplingParams(max_tokens=4)))
    batch = sched.schedule()
    assert len(batch.seqs) == 2  # 200 + 200 <= 512 < 600
    assert sum(s.num_tokens for s in batch.seqs) <= 512


def test_decode_batch_contains_all_running():
    sched, _ = mksched(max_num_seqs=8)
    for i in range(5):
        sched.add(Sequence([i + 1] * 7, SamplingParams(max_tokens=10)))
    b = sched.schedule()
    sched.postprocess(b, [[fake_next(s)] for s in b.seqs])
    b2 = sched.schedule()
    assert b2.kind == BatchKind.DECODE
    assert {s.seq_id for s in b2.seqs} == {s.seq_id for s in b.seqs}


def test_preemption_tiny_pool_all_finish():
    # Reference: big pool, no preemption.
    prompts = [[(7 * i + j) % 50 + 1 for j in range(20 + 3 * i)] for i in range(6)]

    def outputs(num_blocks):
        sched, _ = mksched(num_blocks=num_blocks, enable_prefix_cache=False)
        seqs = [Sequence(p, SamplingParams(max_tokens=40, ignore_eos=True)) for p in prompts]
        for s in seqs:
            sched.add(s)
        fin, _ = drive(sched)
        assert len(fin) == len(prompts)
        return [s.completion_token_ids for s in seqs], sched.num_preemptions, seqs

    ref, n0, _ = outputs(200)
    assert n0 == 0
    got, npre, seqs = outputs(8)
    assert npre > 0
    assert got == ref
    assert all(s.status == SequenceStatus.FINISHED for s in seqs)


def test_preempted_goes_to_front_of_waiting():
    sched, bm = mksched(num_blocks=4, enable_prefix_cache=False)
    a = Sequence([1] * 16, SamplingParams(max_tokens=40, ignore_eos=True))
    b = Sequence([2] * 16, SamplingParams(max_tokens=40, ignore_eos=True))
    c = Sequence([3] * 5, SamplingParams(max_tokens=40, ignore_eos=True))
    for s in (a, b):
        sched.add(s)
    pb = sched.schedule()  # prefill a, b: 1 block each
    sched.postprocess(pb, [[5], [5]])  # both now have 17 tokens -> need 2nd block at decode
    sched.add(c)
    pc = sched.schedule()  # prefill c: 1 block -> 1 block left
    assert pc.seqs == [c]
    sched.postprocess(pc, [[5]])
    d = sched.schedule()  # a takes the last block; b needs one -> preempt c (most recent)
    assert d.kind == BatchKind.DECODE
    assert sched.num_preemptions >= 1
    assert sched.waiting[0] is c
    assert c.status == SequenceStatus.WAITING
    bm.check_invariants(sched.running)


def test_eos_and_ignore_eos():
    sched, _ = mksched()
    s1 = Sequence([5] * 4, SamplingParams(max_tokens=10))
    s2 = Sequence([5] * 4, SamplingParams(max_tokens=10, ignore_eos=True))
    sched.add(s1)
    sched.add(s2)
    b = sched.schedule()
    fin = sched.postprocess(b, [[EOS], [EOS]])
    assert fin == [s1] and s1.finish_reason == "stop"
    assert s2.status == SequenceStatus.RUNNING


def test_max_tokens_and_stop_ids():
    sched, _ = mksched()
    s1 = Sequence([5] * 4, SamplingParams(max_tokens=2))
    s2 = Sequence([5] * 4, SamplingParams(max_tokens=10, stop_token_ids=[42]))
    sched.add(s1)
    sched.add(s2)
    b = sched.schedule()
    assert sched.postprocess(b, [[7], [8]]) == []
    b = sched.schedule()
    fin = sched.postprocess(b, [[7], [42]])
    assert set(fin) == {s1, s2}
    assert s1.finish_reason == "length" and s1.num_completion_tokens == 2
    assert s2.finish_reason == "stop" and s2.completion_token_ids == [8, 42]
    assert not sched.has_work()


def test_spec_postprocess_multi_token_with_truncation():
    sched, bm = mksched(spec_method="oracle")
    s1 = Sequence([5] * 4, SamplingParams(max_tokens=100))
    s2 = Sequence([5] * 4, SamplingParams(max_tokens=5, ignore_eos=True))
    s3 = Sequence([5] * 4, SamplingParams(max_tokens=100))
    for s in (s1, s2, s3):
        sched.add(s)
    b = sched.schedule()
    sched.postprocess(b, [[9], [9], [9]])
    b = sched.schedule()
    assert b.kind == BatchKind.SPEC
    # lookahead 15 slots reserved past the last token
    for s in b.seqs:
        assert len(s.block_table) * BS >= s.num_tokens + 15
    toks16 = list(range(10, 26))
    fin = sched.postprocess(b, [[11, 12, EOS, 13, 14], list(toks16), [21, 22, 23]])
    # s1 stops at the EOS in the middle of its accepted run
    assert s1.completion_token_ids == [9, 11, 12, EOS] and s1.finish_reason == "stop"
    # s2 hits max_tokens=5 after 4 of 16 tokens
    assert s2.completion_token_ids == [9, 10, 11, 12, 13] and s2.finish_reason == "length"
    assert s3.completion_token_ids == [9, 21, 22, 23] and s3.status == SequenceStatus.RUNNING
    assert set(fin) == {s1, s2}
    bm.check_invariants(sched.running)


def test_reject_request_that_never_fits():
    sched, _ = mksched(num_blocks=2)
    with pytest.raises(ValueError):
        sched.add(Sequence([1] * 10, SamplingParams(max_tokens=100)))


def test_spec_rejects_sampling():
    sched, _ = mksched(spec_method="random")
    with pytest.raises(ValueError):
        sched.add(Sequence([1] * 10, SamplingParams(max_tokens=4, temperature=0.7)))


def test_abort_frees_blocks():
    sched, bm = mksched()
    s = Sequence([1] * 40, SamplingParams(max_tokens=10))
    sched.add(s)
    sched.schedule()
    assert bm.num_used_blocks > 0
    sched.abort(s)
    assert bm.num_used_blocks == 0
    assert not sched.has_work()


def test_prefix_cache_through_scheduler():
    sched, bm = mksched()
    p = list(range(1, 50))
    a = Sequence(p, SamplingParams(max_tokens=3, ignore_eos=True))
    sched.add(a)
    drive(sched)
    b = Sequence(p + [77], SamplingParams(max_tokens=3, ignore_eos=True))
    sched.add(b)
    batch = sched.schedule()
    assert batch.seqs == [b]
    assert b.num_cached_tokens == 48
