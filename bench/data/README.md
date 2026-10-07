# bench/data

Small prompt files produced by `scripts/prepare_data.py` (seed 0).

| file | source | license |
|---|---|---|
| gsm8k.jsonl | first 32 of `openai/gsm8k` (main, test), DFlash eval prompt format | MIT |
| humaneval.jsonl | first 32 of `openai/openai_humaneval` (test), DFlash eval prompt format | MIT |
| mtbench.jsonl | first turn of the first 32 of `HuggingFaceH4/mt_bench_prompts` | Apache-2.0 (per HF card) |
| mixed_prompts.jsonl | 16 of each of the above + 16 short hand-written instructions, shuffled | as above |

W1 (random lengths) and W2 (shared prefix) are synthetic token-id workloads regenerated from seed 0 by `bench/workloads.py`; they are not stored.
