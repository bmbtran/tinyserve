"""M3: continuous batching, preemption and random arrival are exact (float64)."""

import random

import pytest

from tests.util import tiny_engine
from tinyserve.sampling import SamplingParams

SP = SamplingParams(max_tokens=24, ignore_eos=True)


@pytest.fixture(scope="module")
def prompts():
    rng = random.Random(0)
    return [[rng.randrange(512) for _ in range(rng.randrange(5, 201))] for _ in range(24)]


@pytest.fixture(scope="module")
def solo(tiny_model, prompts):
    """Reference: each prompt alone in a fresh engine (no batching, no cache)."""
    path, _ = tiny_model
    eng = tiny_engine(path, enable_prefix_cache=False)
    return [eng.generate([p], SP)[0]["token_ids"] for p in prompts]


def test_a_batched_equals_solo(tiny_model, prompts, solo):
    path, _ = tiny_model
    eng = tiny_engine(path, max_num_seqs=8)
    outs = eng.generate(prompts, SP)
    assert [o["token_ids"] for o in outs] == solo
    assert eng.metrics()["steps"]["DECODE"] > 0
    assert eng.block_manager.num_used_blocks == 0


def test_b_preemption_tiny_pool(tiny_model, prompts, solo):
    path, _ = tiny_model
    # 12 blocks x 20 tokens = 240 slots: the longest request (200 + 24) only
    # just fits alone, so the 8 concurrent ones must preempt constantly.
    eng = tiny_engine(path, max_num_seqs=8, num_kv_blocks=12, block_size=20, enable_prefix_cache=False)
    outs = eng.generate(prompts, SP)
    assert eng.scheduler.num_preemptions > 0
    assert sum(o["num_preemptions"] for o in outs) == eng.scheduler.num_preemptions
    assert all(o["finish_reason"] == "length" for o in outs)
    assert [o["token_ids"] for o in outs] == solo
    assert eng.block_manager.num_used_blocks == 0


def test_b2_preemption_with_prefix_cache(tiny_model, prompts, solo):
    path, _ = tiny_model
    eng = tiny_engine(path, max_num_seqs=8, num_kv_blocks=12, block_size=20, enable_prefix_cache=True)
    outs = eng.generate(prompts, SP)
    assert eng.scheduler.num_preemptions > 0
    assert [o["token_ids"] for o in outs] == solo


def test_c_random_arrival(tiny_model, prompts, solo):
    path, _ = tiny_model
    eng = tiny_engine(path, max_num_seqs=8)
    rng = random.Random(1)
    pending = list(enumerate(prompts))
    seqs = {}
    while pending or eng.has_work():
        for _ in range(rng.randrange(0, 4)):
            if pending:
                i, p = pending.pop(0)
                seqs[i] = eng.add_request(p, SP)
        if eng.has_work():
            eng.step()
    assert [seqs[i].completion_token_ids for i in range(len(prompts))] == solo


def test_mixed_sampling_params_in_one_batch(tiny_model, prompts, solo):
    path, _ = tiny_model
    eng = tiny_engine(path, max_num_seqs=8)
    params = [SP if i % 2 == 0 else SamplingParams(max_tokens=24, temperature=1.0, ignore_eos=True, seed=i)
              for i in range(8)]
    outs = eng.generate(prompts[:8], params)
    for i in range(0, 8, 2):  # greedy rows unaffected by sampled neighbours
        assert outs[i]["token_ids"] == solo[i]
