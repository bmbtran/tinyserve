"""Real Qwen3-0.6B in float32 on CPU: greedy tokens identical to HF generate."""

import pytest
import torch

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Explain photosynthesis in one sentence.",
    "1, 2, 3, 4,",
]


@pytest.mark.slow
def test_qwen3_06b_greedy_matches_hf():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tinyserve.config import EngineConfig
    from tinyserve.engine import LLMEngine
    from tinyserve.loader import resolve_model_path
    from tinyserve.sampling import SamplingParams

    path = resolve_model_path("Qwen/Qwen3-0.6B")
    tok = AutoTokenizer.from_pretrained(path)
    hf = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, attn_implementation="sdpa").eval()
    prompts = [tok.encode(p) for p in PROMPTS]
    refs = []
    with torch.no_grad():
        for ids in prompts:
            out = hf.generate(torch.tensor([ids]), max_new_tokens=24, do_sample=False, temperature=None, top_p=None, top_k=None)
            refs.append(out[0, len(ids):].tolist())
    del hf

    eng = LLMEngine(EngineConfig(model=str(path), device="cpu", dtype="float32", attn_backend="torch", block_size=16,
                                 max_model_len=256, max_num_seqs=4, max_num_batched_tokens=1024))
    outs = eng.generate(prompts, SamplingParams(max_tokens=24))
    for p, ref, o in zip(PROMPTS, refs, outs):
        print(f"{p!r} -> {o['text']!r}")
        assert o["token_ids"] == ref, (p, ref, o["token_ids"])
