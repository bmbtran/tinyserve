"""M4: prefix caching is exact, hits the expected number of tokens, and frees blocks."""

import random

from tests.util import tiny_engine
from tinyserve.sampling import SamplingParams

SP = SamplingParams(max_tokens=16, ignore_eos=True)
BS = 16


def _requests(n=10, prefix_len=100, suffix_len=10, seed=0):
    rng = random.Random(seed)
    prefix = [rng.randrange(512) for _ in range(prefix_len)]
    return [prefix + [rng.randrange(512) for _ in range(suffix_len)] for _ in range(n)]


def _run(eng, reqs):
    """First request alone (so its blocks get committed), then the rest together."""
    first = eng.generate(reqs[:1], SP)
    return [o["token_ids"] for o in first + eng.generate(reqs[1:], SP)]


def test_cache_on_equals_off_and_hit_rate(tiny_model):
    path, _ = tiny_model
    reqs = _requests()
    off = _run(tiny_engine(path, enable_prefix_cache=False), reqs)
    eng = tiny_engine(path, enable_prefix_cache=True)
    on = _run(eng, reqs)
    assert on == off
    m = eng.metrics()
    # Block math: 100 shared tokens = 6 full blocks of 16 (96 tokens); the 7th
    # block mixes prefix and suffix, so never hits. 9 of 10 requests hit 96 each.
    expected = 9 * (100 // BS) * BS / (10 * 110)
    assert m["prefix_hit_tokens"] == 9 * 96
    assert m["prefix_query_tokens"] == 10 * 110
    assert abs(m["prefix_hit_rate"] - expected) < 1e-12
    assert m["prefix_hit_rate"] >= 0.75
    # every block's refcount is back to 0
    assert all(b.ref_count == 0 for b in eng.block_manager.blocks)
    assert eng.block_manager.num_free_blocks == eng.block_manager.num_blocks


def test_identical_prompts_whole_prompt_cached(tiny_model):
    path, _ = tiny_model
    p = list(range(1, 65))  # exactly 4 blocks
    ref = tiny_engine(path, enable_prefix_cache=False).generate([p], SP)[0]["token_ids"]
    eng = tiny_engine(path)
    outs = [eng.generate([p], SP)[0]["token_ids"] for _ in range(3)]
    assert outs == [ref] * 3
    # 2nd and 3rd: last block recomputed (need >= 1 token for logits) -> 48 hits each
    assert eng.metrics()["prefix_hit_tokens"] == 2 * 48


def test_generated_tokens_become_cacheable(tiny_model):
    """Blocks filled during decode are committed too: a follow-up turn that
    re-sends prompt + answer hits them (multi-turn chat)."""
    path, _ = tiny_model
    eng = tiny_engine(path)
    p = list(range(1, 41))
    a = eng.generate([p], SamplingParams(max_tokens=40, ignore_eos=True))[0]["token_ids"]
    follow = p + a + [7, 7, 7]
    out = eng.generate([follow], SP)[0]["token_ids"]
    # 80 tokens, but the last sampled token never ran through the model: KV
    # exists for 79 positions -> 4 full blocks.
    assert eng.metrics()["prefix_hit_tokens"] == ((80 - 1) // BS) * BS
    ref = tiny_engine(path, enable_prefix_cache=False).generate([follow], SP)[0]["token_ids"]
    assert out == ref
