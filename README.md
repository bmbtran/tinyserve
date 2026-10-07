# tinyserve

(Work in progress; the full write-up is generated last, from artifacts only.)

## What didn't work / didn't reproduce

### M2: bf16 GPU thresholds (FAIL as specified; analysis)

Artifacts: `results/verify/m2.json`, `results/verify/m2.log`.

* **(a) flash vs torch backend, prefill logits max-abs-diff < 0.15: measured 0.75 (FAIL).**
  Top-1 agreement was 99.55% (passes the >= 99% part). Same-run diagnostic: each bf16 backend is
  about as far from a float32 run of the same weights (flash 0.89, torch-bf16 0.77 max-abs), and
  flash agrees with fp32 on top-1 *more* often (99.55%) than the torch backend does (99.11%).
  Mean |flash - torch| is 0.04. Conclusion: the 0.75 gap is bf16 rounding in a 28-layer bf16
  residual stream (logits are ~20-30, where one bf16 ulp is 0.125), not a backend bug. A 0.15
  max-abs threshold is below the bf16 noise floor for this model. The exactness of the math is
  established on CPU instead (float64 tiny model: < 1e-9 vs HF; real Qwen3-0.6B float32:
  token-identical to HF `generate`).
* **(c) greedy 128 tokens vs HF bf16, >= 7/8 exact: measured 3/8 exact, 8/8 exact-or-near-tie (FAIL on the exact count).**
  Every divergence happened where HF's own top-1/top-2 gap was 0.0-0.125 (0-1 bf16 ulp). The
  same-run baseline is decisive: **HF bf16 sdpa vs HF bf16 eager (same weights, same framework)
  is also only 3/8 exact, 8/8 near-tie.** Exact agreement over 128 bf16 greedy tokens is not a
  property any two attention kernels have here, so the >= 7/8 target was unrealistic. Next time:
  compare at >= 8 decimal precision (fp32 on GPU) for exactness and keep bf16 for the near-tie rule only.

### M3: batched vs sequential in bf16 (FAIL on the exact count; analysis)

Artifacts: `results/verify/m3.json`, `results/verify/m3.log`.

* **Batched (max_num_seqs=32) vs bs=1 sequential, 64 chat prompts x 128 tokens: 24/64 exact (target >= 60), 64/64 exact-or-near-tie.**
  All 40 divergences happened where the bs=1 run's own top-1/top-2 logit gap was <= 0.25, and
  **20 of them at a gap of exactly 0.0** (two vocabulary entries with the same bf16 logit). With a
  batch of 32 instead of 1, cuBLAS picks different GEMM kernels, the bf16 hidden states differ in
  the last bit, and exact ties flip. The same code is exact in float64 on CPU (`tests/cpu/test_engine_batching.py`:
  batched, preempted, random-arrival == solo, token for token), so this is bf16 arithmetic, not
  scheduling or paging. The >= 60/64 target assumed bf16 greedy decoding is far more stable than it is
  for 128-token outputs: 40/64 outputs reach a near-tie within 128 tokens, 20/64 an exact tie.
