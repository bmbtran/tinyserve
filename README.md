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

### M4: prefix cache on vs off in bf16 (hit rate and TTFT pass; exact count FAILs)

Artifacts: `results/verify/m4.json`, `results/verify/m4.log`.

* Hit rate **0.943** (target >= 0.85, PASS) and TTFT p50 **553.7 ms -> 56.1 ms** (ratio 0.101, target <= 0.5, PASS: 9.9x faster).
* **Outputs cache-on vs cache-off: 14/32 exact (target >= 31), 32/32 exact-or-near-tie (FAIL on the exact count).**
  With the cache on, the 2048-token prefix is not recomputed: the 64 suffix tokens attend to it
  through `flash_attn_varlen_func(block_table=...)` instead of in one fresh causal pass, so the bf16
  rounding differs. The prompts are random token ids, which give the model flat next-token
  distributions, so near-ties are even more common than with real text. Every divergence was a near-tie (< 0.5 logit gap).
  The CPU float64 test shows cache-on == cache-off exactly (`tests/cpu/test_prefix_cache.py`).
  (This run did not store per-divergence gaps; later suites do.)

### M7: DFlash (speedup and reference parity pass; MT-Bench tau and the bf16 exact count FAIL)

Artifacts: `results/verify/m7.json`, `results/verify/m7.log`. Qwen3-4B + `z-lab/Qwen3-4B-DFlash-b16`, bf16, L4, greedy, max 512 tokens, thinking disabled.

* **(a) tau parity with the z-lab reference (PASS):** tinyserve tau at bs=1 is within 2% of the vendored
  reference `spec_generate` on the same 8 prompts per dataset (GSM8K 5.65 vs 5.54, HumanEval 6.08 vs 5.96,
  MT-Bench 2.08 vs 2.05). On CPU in float64 the per-step acceptance sequence *and* the draft logits match the
  reference exactly (`tests/cpu/test_dflash_spec.py`, mutation-checked: injected RoPE-offset and missing-k_norm bugs make it fail).
* **(b) tau vs paper: GSM8K 6.02 (0.92x of 6.53) and HumanEval 6.15 (0.93x of 6.64) pass; MT-Bench 2.72 vs 4.35 (0.62x) FAILS the >= 3.0 target.**
  The reference implementation itself gets only 2.05 on our MT-Bench prompts, so this is not a tinyserve bug.
  Likely causes, none verified: we use only the first turn of each MT-Bench conversation and cap output
  at 512 tokens (the paper uses 2048, and longer generations tend to accept more); the paper's exact MT-Bench prompt
  formatting may differ. GSM8K/HumanEval are below the paper by ~7%, consistent with the 512-token cap.
* **(c) lossless spec vs non-spec in bf16: 32/96 exact (target >= 90), 93/96 exact-or-near-tie (target 96/96): FAIL.**
  61 of 64 divergences happen at a top-1/top-2 gap of 0, 0.125 or 0.25 (bf16-quantized logits). The other 3 diverge at a gap of
  **exactly 0.5**, right at the rule's boundary (near-tie means < 0.5), so by the pre-declared rule they are mismatches. The verify
  step runs a 16-token causal varlen attention against the paged cache while non-spec decode runs `flash_attn_with_kvcache`
  one token at a time, so their bf16 rounding differs, the same mechanism as M3/M4. The float64 CPU tests show DFlash spec
  output == non-spec output token for token (oracle, random and real-draft proposers, with preemption and prefix caching).
  Not done (the run did not save which prompts diverged): re-run those 3 prompts in float32 on the GPU (torch backend) to show they converge.
* **(d) speedup (PASS): 3.12x at bs=1** (85.1 vs 27.3 decode tok/s on GSM8K; non-spec uses CUDA graphs, DFlash runs eager),
  3.23x at c=4 and 2.98x at c=16. Below the paper's 5.15x on H200, as expected for an eager spec path on a 300 GB/s card.
