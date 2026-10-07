"""Tiny random Qwen3 in float64: tinyserve must match HF transformers exactly."""

import random

import torch

from tests.util import hf_greedy, tiny_engine
from tinyserve.attention import AttnMetadata, set_attn_metadata
from tinyserve.sampling import SamplingParams


def _full_forward(model, ids: list[int]):
    n = len(ids)
    set_attn_metadata(AttnMetadata(is_varlen=True, cu_seqlens_q=torch.tensor([0, n], dtype=torch.int32),
                                   cu_seqlens_k=torch.tensor([0, n], dtype=torch.int32), max_seqlen_q=n,
                                   max_seqlen_k=n, q_lens=[n], k_lens=[n]))
    try:
        with torch.no_grad():
            return model(torch.tensor(ids), torch.arange(n))
    finally:
        set_attn_metadata(None)


def test_prefill_logits_match_hf(tiny_model):
    path, hf = tiny_model
    eng = tiny_engine(path)
    g = torch.Generator().manual_seed(1)
    for n in (1, 7, 40, 129):
        ids = torch.randint(0, 512, (n,), generator=g).tolist()
        hidden, _ = _full_forward(eng.runner.model, ids)
        logits = eng.runner.model.compute_logits(hidden)
        with torch.no_grad():
            ref = hf(torch.tensor([ids])).logits[0]
        diff = (logits - ref).abs().max().item()
        assert diff < 1e-9, (n, diff)


def test_aux_hidden_is_layer_output(tiny_model):
    from transformers import AutoConfig

    from tinyserve.loader import load_weights
    from tinyserve.models.qwen3 import Qwen3ForCausalLM

    path, hf = tiny_model
    torch.set_default_dtype(torch.float64)
    try:
        m = Qwen3ForCausalLM(AutoConfig.from_pretrained(path), aux_layer_ids=[0], dtype=torch.float64)
    finally:
        torch.set_default_dtype(torch.float32)
    load_weights(m, path)
    ids = list(range(3, 30))
    _, aux = _full_forward(m, ids)
    with torch.no_grad():
        hs = hf(torch.tensor([ids]), output_hidden_states=True).hidden_states
    assert aux.shape == (len(ids), 64)
    assert (aux - hs[1][0]).abs().max().item() < 1e-9


def test_greedy_32_tokens_match_hf(tiny_model):
    path, hf = tiny_model
    eng = tiny_engine(path, block_size=16)
    rng = random.Random(0)
    prompts = [[rng.randrange(512) for _ in range(rng.randrange(5, 60))] for _ in range(8)]
    outs = eng.generate(prompts, SamplingParams(max_tokens=32, ignore_eos=True))
    for p, o in zip(prompts, outs):
        assert o["token_ids"] == hf_greedy(hf, p, 32)


def test_greedy_matches_hf_odd_block_size(tiny_model):
    path, hf = tiny_model
    eng = tiny_engine(path, block_size=5)
    p = list(range(10, 33))
    assert eng.generate([p], SamplingParams(max_tokens=20, ignore_eos=True))[0]["token_ids"] == hf_greedy(hf, p, 20)


def test_sampling_runs_and_is_seeded(tiny_model):
    path, _ = tiny_model
    eng = tiny_engine(path)
    sp = SamplingParams(max_tokens=16, temperature=0.8, top_p=0.9, top_k=50, ignore_eos=True, seed=7)
    a = eng.generate([[1, 2, 3]], sp)[0]["token_ids"]
    b = eng.generate([[1, 2, 3]], sp)[0]["token_ids"]
    assert a == b and len(a) == 16
