# FIXPLAN: resolving the targets missed in the first build

Written 2026-10-07 after the first build (see RESULTS.md / README "What didn't work").
Rule: the original FAIL records stay in RESULTS.md untouched. Where a test was measuring the wrong
thing, the replacement test is added *explicitly* here (with its threshold fixed before running),
not by editing the old one. Budget for this plan: <= $6 extra Modal spend; stop if month-to-date > $12.

## Missed targets

| # | Target (PLAN.md) | Measured | Diagnosis so far |
|---|---|---|---|
| A | M7c lossless spec vs non-spec, 96/96 exact-or-near-tie | 93/96 (3 divergences at gap exactly 0.5) | probably bf16 rounding, **unverified** |
| B | M2a/M2c, M3a, M4a bf16 exact-match counts | 3/8, 24/64, 14/32 (all near-tie) | bf16 kernels differ in the last bit; HF disagrees with itself 5/8. Test measured the wrong thing |
| C | M7b MT-Bench tau >= 3.0 | 2.72 | our protocol differs from the authors' harness (turn 1 only, 512 tokens; theirs: both turns, 2048) |
| D | M8 stretch: throughput >= 85% of vLLM at c=64 | 68.4% | profile: ~21% server/GIL, unfused projections, prefill-first stalls |

## Steps (in order; code changes first so every verification runs on the final code)

### D. Throughput (code changes)
1. **D1 Fused projections.** One `qkv_proj` GEMM instead of q/k/v, one `gate_up_proj` instead of gate/up
   (PLAN.md 4.1 allowed this). The loader maps checkpoint names into slices of the fused weights.
2. **D2 Engine core in its own process.** `server/engine_process.py`: the engine runs in a spawned
   process; the HTTP process only tokenizes/detokenizes. One message per step carries all sequences'
   new tokens. Default for `python -m tinyserve.server.api`; the thread mode stays for CPU tests.
3. **D3 Verify on CPU:** full CPU suite still passes (float64 exactness vs HF with fused weights); new tests
   for the loader mapping and for the process-mode server (stream == non-stream, concurrency, abort).
4. **D4 Verify on GPU:** re-run M5 (graphs) and M6 (server + openai client) suites; then the tinyserve
   `core` benchmark (vLLM numbers from the same day stay as the baseline).
5. **D5 Only if D4 < 85% and budget allows:** decide on chunked prefill from the new profile. Otherwise
   record the remaining gap with the profile evidence.

### B. Replace the bf16 exact-match tests with float32 exactness tests (GPU, Qwen3-0.6B, torch backend)
6. **B1** `tests/gpu/test_fp32_gpu.py`, one container, thresholds fixed now:
   - **B1a (M2c')** tinyserve fp32 vs HF fp32 (sdpa) greedy 128 tokens x 8 prompts: >= 7/8 exact, 8/8 exact-or-fp32-near-tie.
   - **B1b (M2a')** fp32 prefill logits tinyserve vs HF: max-abs < 1e-3. Plus the bf16 kernel check restated
     against fp32 ground truth: flash-bf16 max-abs error vs fp32 <= 1.25x the torch-bf16 backend's error, and
     flash top-1 agreement with fp32 >= 99%. *(This 1.25x criterion is set after seeing M2's 0.89 vs 0.77; it
     is labelled post-hoc in the README.)*
   - **B1c (M3')** batched (32) vs sequential fp32, 64 prompts x 128: >= 60/64 exact, rest fp32-near-tie.
   - **B1d (M4')** prefix cache on vs off fp32, 32 requests (2048 shared + 64): >= 31/32 exact, rest fp32-near-tie.
   - fp32-near-tie = reference top-1/top-2 gap < 0.01 (50x tighter than the bf16 rule).

### A. Settle the three M7c divergences (GPU, Qwen3-4B + DFlash)
7. **A1** `tests/gpu/test_lossless_gpu.py`, one container:
   - bf16, current code: spec vs non-spec on all 96 prompts, **saving prompt indices**, both token streams and gaps.
   - For every bf16 *mismatch* plus the first 8 near-ties: float32 target + float32 draft (torch backend)
     spec vs non-spec on the same prompts, 512 tokens.
   - For each bf16 divergence: the float32 model's own top-1/top-2 and gap at that position.
   - **PASS** iff every fp32 pair is exact (or diverges only at an fp32 gap < 0.01, each listed), and the new
     bf16 run is 96/96 exact-or-near-tie or every bf16 mismatch converges in fp32. If any fp32 pair diverges at
     a gap >= 0.01, that is a bug: stop, debug on CPU, fix.

### C. MT-Bench with the authors' protocol
8. **C1** `tests/gpu/test_mtbench_gpu.py`: tinyserve DFlash, 32 MT-Bench conversations, three protocols:
   (i) turn 1 / 512 tokens (ours, reproduces 2.72), (ii) turn 1 / 2048, (iii) **both turns / 2048** (z-lab harness:
   turn 2 conditioned on the model's own turn-1 answer, thinking disabled). **PASS** iff tau(iii) >= 3.0;
   either way the table attributes the gap to turns vs length.

### Wrap-up
9. Re-run tinyserve `spec` + `spec_dflash` benchmarks if D changed the model code (so W3 charts use final code).
10. Regenerate charts/summary; update README (results + "What didn't work"), LEARNING numbers, RESULTS entries.
11. Verify the Definition of Done below item by item; final report lists each with its artifact.

## Definition of Done (each item verified against an artifact)

- [ ] D1-D3: `uv run pytest tests/cpu -q -m "not slow"` and `-m slow` pass; new loader + process-server tests included (RESULTS.md entry).
- [ ] D4: `results/verify/m5.json` and `m6.json` re-run with status PASS on the new code.
- [ ] D4: new `results/bench/tinyserve_Qwen3-0.6B_W1_*.json`; README states the new % of vLLM at c=64 and whether >= 85% was reached.
- [ ] D5: decision recorded (done, or not done with reason).
- [ ] B1: `results/verify/fp32.json` with status PASS/FAIL per sub-test and the numbers above.
- [ ] A1: `results/verify/lossless.json`: per-divergence table (prompt, position, bf16 gap, fp32 gap, fp32 spec==non-spec); PASS/FAIL per the rule above.
- [ ] C1: `results/verify/mtbench.json`: tau for (i), (ii), (iii); PASS iff tau(iii) >= 3.0.
- [ ] Charts and `results/bench/summary.md` regenerated; README headline/results/"What didn't work" updated from artifacts only; old FAIL records still present in RESULTS.md.
- [ ] COSTLOG.md has a row for every new GPU run; month-to-date <= $12; `modal app list` shows nothing running.
- [ ] `git status` clean; focused commits.
