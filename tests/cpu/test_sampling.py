import math

import pytest
import torch

from tinyserve.sampling import SamplingParams, filter_top_k_top_p, sample


def test_temperature_zero_is_argmax():
    torch.manual_seed(0)
    logits = torch.randn(8, 100)
    out = sample(logits, [SamplingParams()] * 8)
    assert out == logits.argmax(-1).tolist()


def test_top_k_one_is_argmax():
    torch.manual_seed(1)
    logits = torch.randn(6, 50)
    params = [SamplingParams(temperature=1.5, top_k=1)] * 6
    for _ in range(5):
        assert sample(logits, params) == logits.argmax(-1).tolist()


def test_top_p_mass_property():
    probs = torch.tensor([0.5, 0.2, 0.15, 0.1, 0.05])
    logits = probs.log()
    kept = lambda p: torch.isfinite(filter_top_k_top_p(logits, -1, p)).nonzero().flatten().tolist()
    assert kept(0.5) == [0]  # top-1 alone reaches 0.5
    assert kept(0.6) == [0, 1]  # 0.5 < 0.6 <= 0.7
    assert kept(0.7) == [0, 1]
    assert kept(0.71) == [0, 1, 2]
    assert kept(0.96) == [0, 1, 2, 3, 4]
    assert kept(0.01) == [0]  # always keep top-1
    # the kept set is the smallest prefix with mass >= p
    for p in (0.3, 0.55, 0.8, 0.9):
        k = kept(p)
        assert probs[k].sum() >= p - 1e-6
        assert probs[k[:-1]].sum() < p


def test_top_k_filter():
    logits = torch.tensor([1.0, 5.0, 3.0, 4.0])
    out = filter_top_k_top_p(logits, 2, 1.0)
    assert torch.isfinite(out).tolist() == [False, True, False, True]


def test_seeded_determinism():
    logits = torch.randn(4, 200)
    params = [SamplingParams(temperature=1.0, top_p=0.9)] * 4

    def run(seed):
        gens = [torch.Generator().manual_seed(seed + i) for i in range(4)]
        return [sample(logits, params, gens) for _ in range(3)]

    assert run(123) == run(123)
    assert run(123) != run(456)


def test_sampling_distribution_roughly_matches():
    logits = torch.tensor([[math.log(0.7), math.log(0.3)]])
    gen = torch.Generator().manual_seed(0)
    counts = [0, 0]
    for _ in range(2000):
        counts[sample(logits, [SamplingParams(temperature=1.0)], [gen])[0]] += 1
    assert 0.65 < counts[0] / 2000 < 0.75


def test_params_validation():
    with pytest.raises(ValueError):
        SamplingParams(max_tokens=0)
    with pytest.raises(ValueError):
        SamplingParams(top_p=0.0)
