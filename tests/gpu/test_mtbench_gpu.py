"""FIXPLAN C1: MT-Bench tau under three protocols (Qwen3-4B + DFlash, bf16, greedy, thinking off).

  (i)   turn 1 only, max 512 tokens         -- our original M7 protocol (tau 2.72)
  (ii)  turn 1 only, max 2048 tokens
  (iii) both turns, max 2048 tokens         -- the z-lab harness (dflash/benchmark.py):
        turn 2 is conditioned on the model's own decoded turn-1 answer
PASS iff tau(iii) >= 3.0 (PLAN.md M7b threshold). tau is pooled over all spec steps, as in the harness.
"""

import json
from pathlib import Path

import pytest

from bench.workloads import DATA
from tests.util import closed_loop
from tinyserve.config import EngineConfig
from tinyserve.engine import LLMEngine
from tinyserve.sampling import SamplingParams

pytestmark = pytest.mark.gpu


def tau(seqs):
    steps = sum(s.spec_steps for s in seqs)
    return (sum(s.spec_accepted for s in seqs) + steps) / steps, steps


def test_mtbench_protocols(report):
    rows = [json.loads(l) for l in Path(DATA / "mtbench_2turn.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    eng = LLMEngine(EngineConfig(model="Qwen/Qwen3-4B", dtype="bfloat16", device="cuda", attn_backend="flash", block_size=256,
                                 max_num_seqs=16, max_model_len=4096, max_num_batched_tokens=8192, spec_method="dflash",
                                 spec_draft_model="z-lab/Qwen3-4B-DFlash-b16", cuda_graph_batch_sizes=(1, 2, 4, 8, 16)))
    tok = eng.tokenizer
    chat = lambda msgs: tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)  # noqa: E731
    turn1 = [chat([{"role": "user", "content": r["turns"][0]}]) for r in rows]
    eng.generate([turn1[0][:64]], SamplingParams(max_tokens=32))  # warmup

    res = {}
    s512, _ = closed_loop(eng, turn1, 16, SamplingParams(max_tokens=512))
    res["i_turn1_512"] = tau(s512)
    s2048, _ = closed_loop(eng, turn1, 16, SamplingParams(max_tokens=2048))
    res["ii_turn1_2048"] = tau(s2048)
    turn2 = []
    for r, s in zip(rows, s2048):
        answer = tok.decode(s.completion_token_ids, skip_special_tokens=True)
        turn2.append(chat([{"role": "user", "content": r["turns"][0]}, {"role": "assistant", "content": answer},
                           {"role": "user", "content": r["turns"][1]}]))
    s_t2, _ = closed_loop(eng, turn2, 16, SamplingParams(max_tokens=2048))
    res["turn2_only_2048"] = tau(s_t2)
    res["iii_both_turns_2048"] = tau(s2048 + s_t2)
    report["tau"] = {k: round(v[0], 3) for k, v in res.items()}
    report["spec_steps"] = {k: v[1] for k, v in res.items()}
    report["mean_output_tokens"] = {"turn1_512": sum(s.num_completion_tokens for s in s512) / 32,
                                    "turn1_2048": sum(s.num_completion_tokens for s in s2048) / 32,
                                    "turn2_2048": sum(s.num_completion_tokens for s in s_t2) / 32}
    report["finish_length_frac_turn1_512"] = sum(s.finish_reason == "length" for s in s512) / 32
    for k, (t, n) in res.items():
        print(f"C1 MT-Bench {k}: tau={t:.3f} over {n} steps")
    print(f"C1 mean output tokens {report['mean_output_tokens']}; turn-1 answers cut at 512: {report['finish_length_frac_turn1_512']:.0%}")
    print(f"C1 tau(iii) = {res['iii_both_turns_2048'][0]:.3f} (>= 3.0; paper 4.35)")
    assert res["iii_both_turns_2048"][0] >= 3.0
