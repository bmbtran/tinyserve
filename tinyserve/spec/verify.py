"""Greedy verification for speculative decoding.

The target model scores [t, d_1 .. d_K] in one forward and gives its own
greedy choice p_j at every position. Draft token d_{j+1} is accepted while it
equals p_j. The first mismatch is replaced by the target's token p_a, and if
everything matched we still get p_K for free (the "bonus" token). Because
every emitted token is exactly what greedy decoding of the target would
have produced, the output is identical to non-speculative greedy decoding:
the draft only changes how many tokens we get per target forward.
"""

import torch


def greedy_accept(draft: torch.Tensor, target_argmax: torch.Tensor) -> tuple[torch.Tensor, list[list[int]]]:
    """draft: [B, K] draft tokens d_1..d_K; target_argmax: [B, K+1] target
    greedy tokens p_0..p_K (p_j predicts the token after position j).

    Returns (num_accepted [B], new_tokens) where new_tokens[b] =
    draft[b, :a] + [target_argmax[b, a]], always at least one token.
    """
    B, K = draft.shape
    assert target_argmax.shape == (B, K + 1), (draft.shape, target_argmax.shape)
    match = (draft == target_argmax[:, :K]).to(torch.int64)
    # cumprod turns [1,1,0,1] into [1,1,0,0]; the sum is the accepted prefix length.
    num_accepted = match.cumprod(dim=1).sum(dim=1)
    acc = num_accepted.tolist()
    d = draft.tolist()
    p = target_argmax.tolist()
    new_tokens = [d[b][: acc[b]] + [p[b][acc[b]]] for b in range(B)]
    return num_accepted, new_tokens
