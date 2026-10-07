import torch

from tinyserve.spec.verify import greedy_accept

K = 15


def test_all_accept():
    d = torch.arange(1, K + 1).unsqueeze(0)
    p = torch.arange(1, K + 2).unsqueeze(0)  # p[:K] == d, p[K] is the bonus
    a, new = greedy_accept(d, p)
    assert a.tolist() == [15]
    assert new == [list(range(1, 17))]
    assert len(new[0]) == 16


def test_none_accept():
    d = torch.full((1, K), 5)
    p = torch.full((1, K + 1), 7)
    a, new = greedy_accept(d, p)
    assert a.tolist() == [0]
    assert new == [[7]]


def test_first_mismatch_in_middle():
    d = torch.arange(1, K + 1).unsqueeze(0)
    p = torch.arange(1, K + 2).unsqueeze(0).clone()
    p[0, 6] = 999  # mismatch at draft index 6
    p[0, 9] = 555  # later mismatch must not matter
    a, new = greedy_accept(d, p)
    assert a.tolist() == [6]
    assert new == [[1, 2, 3, 4, 5, 6, 999]]


def test_later_match_after_mismatch_not_counted():
    d = torch.tensor([[1, 2, 3, 4]])
    p = torch.tensor([[1, 9, 3, 4, 5]])
    a, new = greedy_accept(d, p)
    assert a.tolist() == [1]
    assert new == [[1, 9]]


def test_batch_mixed():
    d = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    p = torch.tensor([[1, 2, 3, 4], [0, 2, 3, 4], [1, 2, 0, 4]])
    a, new = greedy_accept(d, p)
    assert a.tolist() == [3, 0, 2]
    assert new == [[1, 2, 3, 4], [0], [1, 2, 0]]
