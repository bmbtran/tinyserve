"""Write the small, committed benchmark prompt files (runs locally, free).

    uv run python scripts/prepare_data.py

bench/data/gsm8k.jsonl, humaneval.jsonl, mtbench.jsonl: 32 prompts each, in
exactly the format the DFlash authors' eval uses (z-lab utils.py
`load_and_process_dataset`), as chat messages. MT-Bench: first turn only.
bench/data/mixed_prompts.jsonl: 64 prompts mixing all three plus short
instructions (M3 batching test).

The synthetic token-id workloads (W1 random lengths, W2 shared prefix) are
not stored: bench/workloads.py regenerates them bit-identically from seed 0.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).resolve().parent.parent / "bench" / "data"
N = 32

GSM8K_FMT = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
HUMANEVAL_FMT = "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```"

SHORT = [
    "Write a haiku about autumn leaves.",
    "What is the capital of Australia?",
    "Give me three tips for a job interview.",
    "Explain what a hash table is to a ten-year-old.",
    "Translate 'good morning, how are you?' into Spanish and French.",
    "Name four planets and one fact about each.",
    "Write a limerick about a cat who loves coffee.",
    "What are the main causes of inflation?",
    "Describe the water cycle in three sentences.",
    "Suggest a name for a bakery that also sells books.",
    "What is the difference between a virus and a bacterium?",
    "Write a short motivational message for a student before exams.",
    "How do I reverse a linked list? Explain briefly.",
    "List the first ten prime numbers.",
    "What does HTTP status code 404 mean?",
    "Give a one-paragraph summary of the French Revolution.",
]


def write(name: str, rows: list[dict]) -> None:
    path = OUT / name
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(rows)} rows, {path.stat().st_size / 1024:.0f} KiB)")


def chat(text: str, source: str, idx) -> dict:
    return {"source": source, "id": idx, "messages": [{"role": "user", "content": text}]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    he = load_dataset("openai/openai_humaneval", split="test")
    mt = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")

    gsm_rows = [chat(GSM8K_FMT.format(question=gsm[i]["question"]), "gsm8k", i) for i in range(N)]
    he_rows = [chat(HUMANEVAL_FMT.format(prompt=he[i]["prompt"]), "humaneval", he[i]["task_id"]) for i in range(N)]
    mt_rows = [chat(mt[i]["prompt"][0], "mtbench", int(mt[i]["prompt_id"])) for i in range(N)]
    write("gsm8k.jsonl", gsm_rows)
    write("humaneval.jsonl", he_rows)
    write("mtbench.jsonl", mt_rows)

    rng = random.Random(0)
    mixed = gsm_rows[:16] + he_rows[:16] + mt_rows[:16] + [chat(t, "short", i) for i, t in enumerate(SHORT)]
    rng.shuffle(mixed)
    write("mixed_prompts.jsonl", mixed)

    (OUT / "README.md").write_text(
        "# bench/data\n\n"
        "Small prompt files produced by `scripts/prepare_data.py` (seed 0).\n\n"
        "| file | source | license |\n|---|---|---|\n"
        "| gsm8k.jsonl | first 32 of `openai/gsm8k` (main, test), DFlash eval prompt format | MIT |\n"
        "| humaneval.jsonl | first 32 of `openai/openai_humaneval` (test), DFlash eval prompt format | MIT |\n"
        "| mtbench.jsonl | first turn of the first 32 of `HuggingFaceH4/mt_bench_prompts` | Apache-2.0 (per HF card) |\n"
        "| mixed_prompts.jsonl | 16 of each of the above + 16 short hand-written instructions, shuffled | as above |\n\n"
        "W1 (random lengths) and W2 (shared prefix) are synthetic token-id workloads regenerated from seed 0 by "
        "`bench/workloads.py`; they are not stored.\n",
        encoding="utf-8", newline="\n",
    )


if __name__ == "__main__":
    main()
