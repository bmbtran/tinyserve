# RESULTS

Raw, verified outputs appended by `scripts/record_result.py`. Do not edit entries by hand.

## M1 — 2026-10-07 01:23:43 UTC — git `b3e61ac`

Command: `uv run pytest tests/cpu/test_block_manager.py tests/cpu/test_scheduler.py tests/cpu/test_verify.py tests/cpu/test_sampling.py -q && uv run python -c "import ast,sys; [sys.exit(1) for f in ['tinyserve/block_manager.py','tinyserve/scheduler.py'] if 'torch' in open(f).read()]" && echo NO_TORCH_IN_CORE_OK`  (exit code 0)

```text
...............................................                          [100%]
47 passed in 3.92s
NO_TORCH_IN_CORE_OK
```

## M0 — 2026-10-07 01:29:15 UTC — git `bec12e9` (dirty tree)

Command: `M0 checks (python/torch, git toplevel, download_models, smoke)`
Source file: `results/tmp/m0.txt`
Uncommitted files at record time: `tinyserve/model_runner.py` (--allow-dirty)
Note: flash-attn wheel cxx11abiTRUE works (no fallback needed). Padded rows with cache_seqlens=1 and 0 are finite.

```text
$ uv run python -c "import sys, torch; print(sys.version); print(torch.__version__)"
3.12.13 (main, Mar 24 2026, 22:57:53) [MSC v.1944 64 bit (AMD64)]
2.8.0+cpu
$ git rev-parse --show-toplevel
C:/Users/tranb/projects/tinyserve
$ uv run modal run modal_app.py::download_models   (tail)
Qwen/Qwen3-4B: 8.06 GB in 43s -> /__modal/volumes/vo-S8aOdWLQmEka8BSXxoObJ1/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
z-lab/Qwen3-4B-DFlash-b16: 1.07 GB in 8s -> /__modal/volumes/vo-S8aOdWLQmEka8BSXxoObJ1/hub/models--z-lab--Qwen3-4B-DFlash-b16/snapshots/b74e3a329c4d963783143b1e970d95b002be72bd
$ uv run modal run modal_app.py::smoke
torch=2.8.0+cu128 cuda=12.8 triton=3.4.0
torch._C._GLIBCXX_USE_CXX11_ABI=True
GPU NVIDIA L4 capability=(8, 9)
name, driver_version, memory.total [MiB]
NVIDIA L4, 580.95.05, 23034 MiB
flash_attn=2.8.3.post1 wheel=flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
FA_OK paged_kvcache max_abs_err=9.64e-04 <1e-2 vs torch reference
  varlen causal=True max_abs_err=1.29e-03
  varlen causal=False max_abs_err=1.17e-03
FA_VARLEN_BLOCKTABLE_OK
FA_PAD_ROWS finite=True (cache_seqlens=1, block 0)
FA_PAD_ROWS_ZERO_LEN ok finite_real_row=True
TRITON_OK
weights cached: Qwen/Qwen3-0.6B
weights cached: Qwen/Qwen3-4B
weights cached: z-lab/Qwen3-4B-DFlash-b16
```

## M2 (CPU) — 2026-10-07 01:46:13 UTC — git `19523b5` (dirty tree)

Command: `uv run pytest tests/cpu -q -m "not slow" && uv run pytest tests/cpu -q -m slow`  (exit code 0)
Uncommitted files at record time: `OSTLOG.md`

```text
....................................................                     [100%]
52 passed, 1 deselected in 12.68s
.                                                                        [100%]
1 passed, 52 deselected in 65.27s (0:01:05)
```

## M2 (GPU) — 2026-10-07 01:46:25 UTC — git `5043c36`

Command: `uv run modal run modal_app.py::gpu_tests --suite m2`
Source file: `results/verify/m2.log`
Note: FAIL on thresholds (a) max_abs_diff<0.15 and (c) >=7/8 exact. Diagnostics in the same run: both bf16 backends are ~0.8 max-abs from an fp32 run, and HF bf16 sdpa vs HF bf16 eager is also only 3/8 exact (8/8 near-tie). See README 'What didn't work'. Run at git 19523b5 (test code identical).

```text
M2a diag: |flash-fp32|max=0.8868 |torch_bf16-fp32|max=0.7737 mean|flash-torch|=0.04117
M2a flash-vs-torch max_abs_diff=0.7500 (<0.15) top1_agree=0.9955 (>=0.99)
FM2b triton_store == torch_store: True
.M2c diag: HF-sdpa vs HF-eager (both bf16) exact=3/8
  near_tie diverge_at=12 gap=0.0625
  near_tie diverge_at=78 gap=0.125
  exact diverge_at=None gap=None
  near_tie diverge_at=10 gap=0.0
  exact diverge_at=None gap=None
  exact diverge_at=None gap=None
  near_tie diverge_at=15 gap=0.125
  near_tie diverge_at=97 gap=0.125
M2c greedy vs HF bf16: exact=3/8 (>=7) exact_or_near_tie=8/8 (8)
F
M2 FAIL a_max_abs_diff=0.75 a_top1_agree=0.9955 a_diag_flash_vs_fp32=0.8868 a_diag_torch_bf16_vs_fp32=0.7737 a_diag_flash_top1_vs_fp32=0.9955 a_diag_torch_top1_vs_fp32=0.9911 a_diag_mean_abs_diff=0.04117 b_triton_store_exact=True c_diag_hf_sdpa_vs_hf_eager_exact=3/8 c_diag_hf_sdpa_vs_hf_eager_ok=8/8 c_exact=3/8 c_exact_or_near_tie=8/8

=================================== FAILURES ===================================
________________________ test_a_flash_vs_torch_backend _________________________

flash_engine = <tinyserve.engine.LLMEngine object at 0x2a69bd30bcb0>
report = {'a_max_abs_diff': 0.75, 'a_top1_agree': 0.9955, 'a_diag_flash_vs_fp32': 0.8868, 'a_diag_torch_bf16_vs_fp32': 0.7737, ...}

    def test_a_flash_vs_torch_backend(flash_engine, report):
        tok = flash_engine.tokenizer
        seqs = [tok.encode(p * 6) for p in PROMPTS[:4]]
        lf = full_prefill_logits(flash_engine.runner.model, seqs)
        # Same weights, torch backend: swap the backend module on every layer.
        from tinyserve.attention import backend_torch
    
        for layer in flash_engine.runner.layers:
            layer.backend = backend_torch
        try:
            lt = full_prefill_logits(flash_engine.runner.model, seqs)
        finally:
            from tinyserve.attention import backend_flash
    
            for layer in flash_engine.runner.layers:
                layer.backend = backend_flash
        diff = (lf - lt).abs().max().item()
        top1 = (lf.argmax(-1) == lt.argmax(-1)).float().mean().item()
        report["a_max_abs_diff"] = round(diff, 4)
        report["a_top1_agree"] = round(top1, 4)
    
        # Diagnostic (does not change the threshold): how far is EACH bf16 backend
        # from a float32 run of the same weights? If both are equally far, the
        # flash/torch gap is bf16 noise, not a backend bug.
        from tinyserve.loader import load_weights
        from tinyserve.models.qwen3 import Qwen3ForCausalLM
    
        with torch.device("cuda"):
            m32 = Qwen3ForCausalLM(flash_engine.hf_config, backend="torch", max_positions=4096, dtype=torch.float32)
        load_weights(m32, flash_engine.model_path)
        l32 = full_prefill_logits(m32, seqs)
        del m32
        torch.cuda.empty_cache()
        e_flash = (lf - l32).abs().max().item()
        e_torch = (lt - l32).abs().max().item()
        report["a_diag_flash_vs_fp32"] = round(e_flash, 4)
        report["a_diag_torch_bf16_vs_fp32"] = round(e_torch, 4)
        report["a_diag_flash_top1_vs_fp32"] = round((lf.argmax(-1) == l32.argmax(-1)).float().mean().item(), 4)
        report["a_diag_torch_top1_vs_fp32"] = round((lt.argmax(-1) == l32.argmax(-1)).float().mean().item(), 4)
        report["a_diag_mean_abs_diff"] = round((lf - lt).abs().mean().item(), 5)
        print(f"M2a diag: |flash-fp32|max={e_flash:.4f} |torch_bf16-fp32|max={e_torch:.4f} "
              f"mean|flash-torch|={report['a_diag_mean_abs_diff']}")
        print(f"M2a flash-vs-torch max_abs_diff={diff:.4f} (<0.15) top1_agree={top1:.4f} (>=0.99)")
>       assert diff < 0.15 and top1 >= 0.99
E       assert (0.75 < 0.15)

tests/gpu/test_m2_gpu.py:97: AssertionError
___________________________ test_c_greedy_vs_hf_bf16 ___________________________

flash_engine = <tinyserve.engine.LLMEngine object at 0x2a69bd30bcb0>
report = {'a_max_abs_diff': 0.75, 'a_top1_agree': 0.9955, 'a_diag_flash_vs_fp32': 0.8868, 'a_diag_torch_bf16_vs_fp32': 0.7737, ...}

    def test_c_greedy_vs_hf_bf16(flash_engine, report):
        from transformers import AutoModelForCausalLM
    
        hf = AutoModelForCausalLM.from_pretrained(flash_engine.model_path, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
        tok = flash_engine.tokenizer
        prompts = [tok.encode(p) for p in PROMPTS]
        refs = [hf_greedy(hf, p, 128, return_gaps=True) for p in prompts]
        # Diagnostic baseline: HF against ITSELF with a different attention kernel.
        hf.config._attn_implementation = "eager"
        hf.set_attn_implementation("eager") if hasattr(hf, "set_attn_implementation") else None
        eager = [hf_greedy(hf, p, 128) for p in prompts]
        base = [compare_tokens(r[0], e, r[1])[0] for r, e in zip(refs, eager)]
        report["c_diag_hf_sdpa_vs_hf_eager_exact"] = f"{base.count('exact')}/8"
        report["c_diag_hf_sdpa_vs_hf_eager_ok"] = f"{sum(b != 'mismatch' for b in base)}/8"
        print(f"M2c diag: HF-sdpa vs HF-eager (both bf16) exact={base.count('exact')}/8")
        del hf
        torch.cuda.empty_cache()
        outs = flash_engine.generate(prompts, SamplingParams(max_tokens=128, ignore_eos=True))
        verdicts = []
        for (ref, gaps), o in zip(refs, outs):
            v, idx, gap = compare_tokens(ref, o["token_ids"], gaps)
            verdicts.append({"verdict": v, "diverge_at": idx, "gap": gap})
            print(f"  {v} diverge_at={idx} gap={gap}")
        n_exact = sum(v["verdict"] == "exact" for v in verdicts)
        n_ok = sum(v["verdict"] != "mismatch" for v in verdicts)
        report["c_exact"] = f"{n_exact}/8"
        report["c_exact_or_near_tie"] = f"{n_ok}/8"
        report["c_details"] = verdicts
        print(f"M2c greedy vs HF bf16: exact={n_exact}/8 (>=7) exact_or_near_tie={n_ok}/8 (8)")
>       assert n_exact >= 7 and n_ok == 8
E       assert (3 >= 7)

tests/gpu/test_m2_gpu.py:148: AssertionError
==================================== PASSES ====================================
=========================== short test summary info ============================
PASSED tests/gpu/test_m2_gpu.py::test_b_triton_store_equals_torch_store
FAILED tests/gpu/test_m2_gpu.py::test_a_flash_vs_torch_backend - assert (0.75...
FAILED tests/gpu/test_m2_gpu.py::test_c_greedy_vs_hf_bf16 - assert (3 >= 7)
2 failed, 1 passed in 117.23s (0:01:57)
```

## M3 (CPU) — 2026-10-07 01:56:28 UTC — git `0f1c01a`

Command: `uv run pytest tests/cpu/test_engine_batching.py -q`  (exit code 0)

```text
.....                                                                    [100%]
5 passed in 18.45s
```

## M3 (GPU) — 2026-10-07 01:56:30 UTC — git `0f1c01a`

Command: `uv run modal run modal_app.py::gpu_tests --suite m3`
Source file: `results/verify/m3.log`
Note: M3a FAIL on exact count (24/64 vs >=60); 64/64 exact-or-near-tie. All 40 divergences at reference top1-top2 gap <= 0.25 (20 of them exactly 0.0 = bf16 tie). M3b: 2004 output tok/s eager (no threshold).

```text
KV blocks: 613 x 256 tokens
M3a batched(32) vs sequential: exact=24/64 (>=60) exact_or_near_tie=64/64 (64)
FM3b offline 256 x (in 512 / out 256), eager, max_num_seqs=128: 2004 output tok/s, 6012 total tok/s, wall 32.7s, steps {'PREFILL': 16, 'DECODE': 510, 'SPEC': 0}
.
M3 FAIL a_batched_wall_s=10.76 a_exact=24/64 a_exact_or_near_tie=64/64 b_output_tok_s=2004.0 b_total_tok_s=6012.1 b_wall_s=32.7 b_preemptions=0

=================================== FAILURES ===================================
_________________________ test_a_batched_vs_sequential _________________________

engine = <tinyserve.engine.LLMEngine object at 0x2a690c630fe0>
report = {'a_batched_wall_s': 10.76, 'a_exact': '24/64', 'a_exact_or_near_tie': '64/64', 'a_divergences': [{'i': 1, 'verdict': ..., {'i': 7, 'verdict': 'near_tie', 'at': 56, 'gap': 0.0}, {'i': 8, 'verdict': 'near_tie', 'at': 11, 'gap': 0.125}, ...]}

    def test_a_batched_vs_sequential(engine, report):
        rows = load_chat("mixed_prompts")
        prompts = [engine.apply_chat_template(r["messages"]) for r in rows]
        sp = SamplingParams(max_tokens=128)
        engine.runner.record_gaps = True
        seq_outs = [engine.generate([p], sp)[0] for p in prompts]  # bs = 1, one at a time
        engine.runner.record_gaps = False
        engine.cfg.max_num_seqs = 32
        t = time.perf_counter()
        batched = engine.generate(prompts, sp)
        report["a_batched_wall_s"] = round(time.perf_counter() - t, 2)
        engine.cfg.max_num_seqs = 128
        verdicts = [compare_tokens(s["token_ids"], b["token_ids"], s["logit_gaps"]) for s, b in zip(seq_outs, batched)]
        n_exact = sum(v[0] == "exact" for v in verdicts)
        n_ok = sum(v[0] != "mismatch" for v in verdicts)
        report["a_exact"] = f"{n_exact}/64"
        report["a_exact_or_near_tie"] = f"{n_ok}/64"
        report["a_divergences"] = [{"i": i, "verdict": v[0], "at": v[1], "gap": v[2]} for i, v in enumerate(verdicts) if v[0] != "exact"]
        print(f"M3a batched(32) vs sequential: exact={n_exact}/64 (>=60) exact_or_near_tie={n_ok}/64 (64)")
>       assert n_exact >= 60 and n_ok == 64
E       assert (24 >= 60)

tests/gpu/test_m3_gpu.py:45: AssertionError
==================================== PASSES ====================================
=========================== short test summary info ============================
PASSED tests/gpu/test_m3_gpu.py::test_b_offline_throughput
FAILED tests/gpu/test_m3_gpu.py::test_a_batched_vs_sequential - assert (24 >=...
1 failed, 1 passed in 345.23s (0:05:45)
```

## M4 (CPU) — 2026-10-07 01:59:53 UTC — git `e13b353`

Command: `uv run pytest tests/cpu/test_prefix_cache.py -q`  (exit code 0)

```text
...                                                                      [100%]
3 passed in 10.97s
```

## M4 (GPU) — 2026-10-07 01:59:54 UTC — git `e13b353`

Command: `uv run modal run modal_app.py::gpu_tests --suite m4`
Source file: `results/verify/m4.log`
Note: M4b hit_rate 0.943 (>=0.85) PASS; M4c TTFT ratio 0.101 (<=0.5) PASS, 9.87x; M4a FAIL on exact count 14/32 (>=31), 32/32 exact-or-near-tie.

```text
M4a cache on vs off: exact=14/32 (>=31) exact_or_near_tie=32/32 (32)
M4b hit_rate=0.943 (>=0.85)
M4c TTFT p50 off=553.7ms on=56.1ms ratio=0.101 (<=0.5) ttft_speedup=9.87x
F
M4 FAIL exact=14/32 exact_or_near_tie=32/32 hit_rate=0.9432 ttft_p50_off_ms=553.66 ttft_p50_on_ms=56.11 ttft_ratio=0.101 ttft_speedup=9.87

=================================== FAILURES ===================================
_________________________ test_prefix_cache_on_vs_off __________________________

engine = <tinyserve.engine.LLMEngine object at 0x2b1afd927500>
report = {'exact': '14/32', 'exact_or_near_tie': '32/32', 'hit_rate': 0.9432, 'ttft_p50_off_ms': 553.66, ...}

    def test_prefix_cache_on_vs_off(engine, report):
        reqs = w2_shared_prefix(groups=1, per_group=32, prefix_len=2048, suffix=(64, 64), out=64, seed=0)
        prompts = [r["prompt"] for r in reqs]
        engine.generate([prompts[0][:300]], SamplingParams(max_tokens=4, ignore_eos=True))  # warmup (unrelated tokens)
    
        engine.block_manager.enable_prefix_cache = False
        engine.runner.record_gaps = True
        off = closed_loop(engine, prompts, 8)
        engine.runner.record_gaps = False
    
        engine.block_manager.enable_prefix_cache = True
        engine.block_manager.stats = {"prefix_query_tokens": 0, "prefix_hit_tokens": 0}
        on = closed_loop(engine, prompts, 8)
        hit_rate = engine.block_manager.prefix_hit_rate
    
        off_by_prompt = {tuple(s.token_ids[: s.num_prompt_tokens]): s for s in off}
        verdicts = []
        for s in on:
            ref = off_by_prompt[tuple(s.token_ids[: s.num_prompt_tokens])]
            verdicts.append(compare_tokens(ref.completion_token_ids, s.completion_token_ids, ref.logit_gaps)[0])
        n_exact, n_ok = verdicts.count("exact"), sum(v != "mismatch" for v in verdicts)
    
        ttft = lambda seqs: statistics.median((s.first_token_time - s.arrival_time) * 1000 for s in seqs[1:])  # noqa: E731
        t_off, t_on = ttft(off), ttft(on)
        report.update(exact=f"{n_exact}/32", exact_or_near_tie=f"{n_ok}/32", hit_rate=round(hit_rate, 4),
                      ttft_p50_off_ms=round(t_off, 2), ttft_p50_on_ms=round(t_on, 2), ttft_ratio=round(t_on / t_off, 3),
                      ttft_speedup=round(t_off / t_on, 2))
        print(f"M4a cache on vs off: exact={n_exact}/32 (>=31) exact_or_near_tie={n_ok}/32 (32)")
        print(f"M4b hit_rate={hit_rate:.3f} (>=0.85)")
        print(f"M4c TTFT p50 off={t_off:.1f}ms on={t_on:.1f}ms ratio={t_on / t_off:.3f} (<=0.5) ttft_speedup={t_off / t_on:.2f}x")
        assert hit_rate >= 0.85
        assert t_on <= 0.5 * t_off
>       assert n_exact >= 31 and n_ok == 32
E       assert (14 >= 31)

tests/gpu/test_m4_gpu.py:74: AssertionError
=========================== short test summary info ============================
FAILED tests/gpu/test_m4_gpu.py::test_prefix_cache_on_vs_off - assert (14 >= 31)
1 failed in 43.76s
```

## M5 (GPU) — 2026-10-07 02:01:59 UTC — git `58f0cf1`

Command: `uv run modal run modal_app.py::gpu_tests --suite m5`
Source file: `results/verify/m5.log`
Note: PASS: graph==eager 32/32 exact; bs=1 ITL 7.75ms vs 39.07ms eager (0.198); bs=32 10.18ms vs 40.95ms.

```text
captured graphs for batch sizes [1, 2, 4, 8, 16, 32, 64, 128]
M5a graph vs eager: exact=32/32 (>=31) exact_or_near_tie=32/32 (32)
.  bs=  1: graph 7.75 ms  eager 39.07 ms  speedup 5.04x
  bs=  2: graph 8.09 ms  eager 40.11 ms  speedup 4.96x
  bs=  4: graph 8.18 ms  eager 39.75 ms  speedup 4.86x
  bs=  8: graph 8.60 ms  eager 41.22 ms  speedup 4.79x
  bs= 16: graph 8.89 ms  eager 40.84 ms  speedup 4.59x
  bs= 32: graph 10.18 ms  eager 40.95 ms  speedup 4.02x
  bs= 64: graph 12.61 ms  eager 40.69 ms  speedup 3.23x
  bs=128: graph 17.37 ms  eager 41.32 ms  speedup 2.38x
M5b bs=1 ITL p50 graph/eager=0.198 (<=0.67)
M5c bs=32 step graph=10.177ms eager=40.95ms (graph<=eager)
.
M5 PASS a_exact=32/32 a_exact_or_near_tie=32/32 b_bs1_graph_eager_ratio=0.198 c_bs32_graph_ms=10.177 c_bs32_eager_ms=40.95

==================================== PASSES ====================================
=========================== short test summary info ============================
PASSED tests/gpu/test_m5_gpu.py::test_a_graph_vs_eager_outputs
PASSED tests/gpu/test_m5_gpu.py::test_b_c_itl
2 passed in 51.88s
```

## M6 (CPU) — 2026-10-07 02:09:50 UTC — git `3604694`

Command: `uv run pytest tests/cpu/test_server.py -q`  (exit code 0)

```text
.........                                                                [100%]
============================== warnings summary ===============================
.venv\Lib\site-packages\fastapi\testclient.py:1
  C:\Users\tranb\projects\tinyserve\.venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
9 passed, 1 warning in 18.42s
```

## M6 (GPU) — 2026-10-07 02:09:51 UTC — git `3604694`

Command: `uv run modal run modal_app.py::gpu_tests --suite m6`
Source file: `results/verify/m6.log`
Note: PASS. Includes the official openai client streaming transcript (DoD item).

```text
tinyserve: serving Qwen/Qwen3-0.6B on http://127.0.0.1:8011 (KV blocks: 613)
server healthy after 18s
  openai: chat (stream) -> 'Three primary colors are **red, blue, and yellow**.'
  openai: completions (stream) -> ' Paris. The capital of France is also the capital of the'
  openai: chat (non-stream) -> 'Three primary colors are **red, blue, and yellow**.' usage=CompletionUsage(completion_tokens=14, prompt_tokens=17, total_tokens=31, completion_tokens_details=None, prompt_tokens_details=None)
  openai: models -> ['Qwen/Qwen3-0.6B']
.M6 c=32 x 64 req: errors=0 TTFT p50=459.272ms p99=553.875ms ITL p50=19.047ms p99=36.748ms output_tok_s=1387.23
.
M6 PASS errors=0

==================================== PASSES ====================================
=========================== short test summary info ============================
PASSED tests/gpu/test_m6_gpu.py::test_openai_client_streaming
PASSED tests/gpu/test_m6_gpu.py::test_concurrency_32_x_64
2 passed in 33.77s
```

## M7 (CPU) — 2026-10-07 02:35:09 UTC — git `2a45a7a` (dirty tree)

Command: `uv run pytest tests/cpu/test_dflash_spec.py -q`  (exit code 0)
Uncommitted files at record time: `modal_app.py`, `LEARNING.md`

```text
..........                                                               [100%]
10 passed in 43.79s
```

## M7 (GPU) — 2026-10-07 02:35:10 UTC — git `2a45a7a` (dirty tree)

Command: `uv run modal run modal_app.py::gpu_tests --suite m7`
Source file: `results/verify/m7.log`
Uncommitted files at record time: `modal_app.py`, `LEARNING.md` (--allow-dirty)
Note: Run at git 16b6bbd (dirty files at record time are later bench/doc edits, not used by this run). (a) tau parity PASS (ratios 1.016-1.020); (b) FAIL on MT-Bench only (2.716 < 3.0; gsm8k 6.02, humaneval 6.15 pass); (c) FAIL: 32/96 exact, 93/96 exact-or-near-tie, the 3 others diverge at gap exactly 0.5 (rule is < 0.5); (d) PASS 3.12x at bs=1 (85.1 vs 27.3 tok/s), 3.23x c=4, 2.98x c=16.

```text

Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]
Loading checkpoint shards:  33%|███▎      | 1/3 [00:02<00:05,  2.62s/it]
Loading checkpoint shards:  67%|██████▋   | 2/3 [00:04<00:02,  2.05s/it]
Loading checkpoint shards: 100%|██████████| 3/3 [00:04<00:00,  1.44s/it]
  reference gsm8k: tau=5.541 over 423 steps
  reference humaneval: tau=5.956 over 436 steps
  reference mtbench: tau=2.051 over 979 steps
.  tinyserve bs=1 gsm8k: tau=5.653 decode 85.1 tok/s
  tinyserve bs=1 humaneval: tau=6.076 decode 88.1 tok/s
  tinyserve bs=1 mtbench: tau=2.083 decode 31.7 tok/s
  tinyserve c=16 gsm8k: tau=6.023
  tinyserve c=16 humaneval: tau=6.152
  tinyserve c=16 mtbench: tau=2.716
  tinyserve DFlash gsm8k c=4: 317.6 output tok/s
  tinyserve DFlash gsm8k c=16: 873.3 output tok/s
.  non-spec engine graphs: [1, 2, 4, 8, 16]
  non-spec bs=1 gsm8k decode 27.3 tok/s
  non-spec gsm8k c=4: 98.3 output tok/s
  non-spec gsm8k c=16: 293.5 output tok/s
.M7a gsm8k: tinyserve tau=5.653 reference tau=5.541 ratio=1.020 (within +-10%)
M7a humaneval: tinyserve tau=6.076 reference tau=5.956 ratio=1.020 (within +-10%)
M7a mtbench: tinyserve tau=2.083 reference tau=2.051 ratio=1.016 (within +-10%)
.M7b gsm8k: tau=6.023 paper=6.53 ratio=0.922 (>= 4.5)
M7b humaneval: tau=6.152 paper=6.64 ratio=0.927 (>= 4.6)
M7b mtbench: tau=2.716 paper=4.35 ratio=0.624 (>= 3.0)
FM7c spec vs non-spec: exact=32/96 (>=90) exact_or_near_tie=93/96 (96)
FM7d gsm8k c=1: DFlash 85.1 tok/s vs non-spec (CUDA graphs) 27.3 tok/s -> 3.12x (>= 2.0)
M7d gsm8k c=4: DFlash 317.6 tok/s vs non-spec (CUDA graphs) 98.3 tok/s -> 3.23x
M7d gsm8k c=16: DFlash 873.3 tok/s vs non-spec (CUDA graphs) 293.5 tok/s -> 2.98x
.
M7 FAIL a_diag_identical_outputs_vs_reference=5/24 c_exact=32/96 c_exact_or_near_tie=93/96

=================================== FAILURES ===================================
_____________________________ test_4b_tau_vs_paper _____________________________

report = {'ref_tau': {'gsm8k': 5.541, 'humaneval': 5.956, 'mtbench': 2.051}, 'tinyserve_tau_bs1_first8': {'gsm8k': 5.653, 'huma...mtbench': 2.716}, 'engine_metrics_spec': {'running': 0, 'waiting': 0, 'kv_usage': 0.0, 'num_kv_blocks': 182, ...}, ...}

    def test_4b_tau_vs_paper(report):
        rows, ok = {}, True
        for d in SPEC_DATASETS:
            t, p = S["ts_tau_all"][d], PAPER_TAU_QWEN3_4B[d]
            rows[d] = {"tinyserve": round(t, 3), "paper": p, "ratio": round(t / p, 3), "min": PAPER_MIN[d]}
            ok &= t >= PAPER_MIN[d]
            print(f"M7b {d}: tau={t:.3f} paper={p} ratio={t / p:.3f} (>= {PAPER_MIN[d]})")
        report["b_tau_vs_paper"] = rows
>       assert ok
E       assert False

tests/gpu/test_m7_gpu.py:185: AssertionError
_______________________________ test_4c_lossless _______________________________

report = {'ref_tau': {'gsm8k': 5.541, 'humaneval': 5.956, 'mtbench': 2.051}, 'tinyserve_tau_bs1_first8': {'gsm8k': 5.653, 'huma...mtbench': 2.716}, 'engine_metrics_spec': {'running': 0, 'waiting': 0, 'kv_usage': 0.0, 'num_kv_blocks': 182, ...}, ...}

    def test_4c_lossless(report):
        verdicts = []
        for d in SPEC_DATASETS:
            for spec, (base, gaps) in zip(S["spec_outputs"][d], S["base_outputs"][d]):
                verdicts.append((d,) + compare_tokens(base, spec, gaps))
        n_exact = sum(v[1] == "exact" for v in verdicts)
        n_ok = sum(v[1] != "mismatch" for v in verdicts)
        report["c_exact"], report["c_exact_or_near_tie"] = f"{n_exact}/96", f"{n_ok}/96"
        report["c_divergences"] = [{"dataset": v[0], "verdict": v[1], "at": v[2], "gap": v[3]} for v in verdicts if v[1] != "exact"]
        print(f"M7c spec vs non-spec: exact={n_exact}/96 (>=90) exact_or_near_tie={n_ok}/96 (96)")
>       assert n_exact >= 90 and n_ok == 96
E       assert (32 >= 90)

tests/gpu/test_m7_gpu.py:198: AssertionError
==================================== PASSES ====================================
=========================== short test summary info ============================
PASSED tests/gpu/test_m7_gpu.py::test_1_reference
PASSED tests/gpu/test_m7_gpu.py::test_2_tinyserve_spec
PASSED tests/gpu/test_m7_gpu.py::test_3_tinyserve_nonspec
PASSED tests/gpu/test_m7_gpu.py::test_4a_tau_parity_with_reference
PASSED tests/gpu/test_m7_gpu.py::test_4d_speedup
FAILED tests/gpu/test_m7_gpu.py::test_4b_tau_vs_paper - assert False
FAILED tests/gpu/test_m7_gpu.py::test_4c_lossless - assert (32 >= 90)
2 failed, 5 passed in 709.25s (0:11:49)
```

## M8 profile (GPU) — 2026-10-07 03:00:02 UTC — git `4c8bdc4`

Command: `uv run modal run modal_app.py::gpu_tests --suite profile`
Source file: `results/verify/profile.log`
Note: One-off torch.profiler + step-split run (PLAN R14).

```text
PROFILE offline engine c=64: 1843 output tok/s (server W1 c=64 is in results/bench)
PROFILE PREFILL: 9 steps, mean ms per step {'schedule': 2.52, 'runner': 344.88, 'postprocess': 1.35}, share of wall 17.1%
PROFILE DECODE: 510 steps, mean ms per step {'schedule': 0.1, 'runner': 29.4, 'postprocess': 0.2}, share of wall 82.7%
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                        cudaMemcpyAsync        86.73%     962.726ms        86.73%     962.726ms       2.188ms       0.000us         0.00%       0.000us       0.000us           440  
                                        cudaGraphLaunch         8.59%      95.398ms         9.21%     102.225ms       2.556ms       1.505us         0.00%       1.505us       0.038us            40  
                                           Buffer Flush         2.20%      24.450ms         2.21%      24.500ms       6.125ms       1.280us         0.00%       2.785us       0.696us             4  
                                Activity Buffer Request         0.84%       9.363ms         0.84%       9.363ms       2.341ms       0.448us         0.00%       0.448us       0.112us             4  
                                            aten::copy_         0.27%       2.998ms        87.35%     969.622ms       2.204ms     438.986us         0.04%     439.434us       0.999us           440  
                                               aten::mm         0.26%       2.862ms         0.35%       3.900ms      97.509us      58.031ms         5.48%      58.031ms       1.451ms            40  
                                    aten::empty_strided         0.14%       1.544ms         0.14%       1.544ms       6.431us       0.000us         0.00%       0.000us       0.000us           240  
                                           aten::argmax         0.13%       1.411ms         0.21%       2.370ms      59.259us       3.749ms         0.35%       3.749ms      93.723us            40  
                                  cudaStreamSynchronize         0.12%       1.377ms         0.12%       1.377ms       5.736us       0.000us         0.00%       0.000us       0.000us           240  
                                         aten::_to_copy         0.11%       1.238ms        87.23%     968.316ms       4.035ms       0.000us         0.00%     145.077us       0.604us           240  
                                            aten::empty         0.11%       1.226ms         0.11%       1.226ms       6.131us       0.000us         0.00%       0.000us       0.000us           200  
                                               aten::to         0.09%       1.030ms        87.32%     969.346ms       4.039ms       0.000us         0.00%     145.077us       0.604us           240  
                                         cuLaunchKernel         0.09%     991.531us         0.09%     991.531us      24.788us       0.000us         0.00%       0.000us       0.000us            40  
                                            aten::slice         0.07%     832.213us         0.11%       1.230ms       5.125us       0.000us         0.00%       0.000us       0.000us           240  
                                       aten::as_strided         0.05%     527.168us         0.05%     527.168us       1.647us       0.000us         0.00%       0.000us       0.000us           320  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.110s
Self CUDA time total: 1.058s

.
PROFILE PASS offline_c64_output_tok_s=1842.6

==================================== PASSES ====================================
=========================== short test summary info ============================
PASSED tests/gpu/test_profile_gpu.py::test_profile_w1
1 passed in 81.75s (0:01:21)
```

## M8 — 2026-10-07 03:20:45 UTC — git `fd9fa6f`

Command: `uv run modal run modal_app.py::bench --engine {tinyserve,vllm} --suite {core,spec}; uv run python bench/plot.py`
Source file: `results/bench/summary.md`
Note: tinyserve W1 throughput at c=64 = 68.4% of vLLM (>=60% PASS, stretch 85% missed). DFlash bs=1 speedup via server 2.90x (>=2.0 PASS). tinyserve tau within 0.8% of vLLM-DFlash tau on all three datasets (PASS). 'errors 2' at every W1 point for BOTH engines = 2 requests whose 256 tokens are all special tokens (empty text, HTTP 200).

```text
# Benchmark summary (auto-generated by `bench/plot.py`)

GPU: NVIDIA L4. tinyserve git `2485804`, vLLM `0.30.0`. Raw data: `results/bench/*.json`.

## W1 (random lengths, Qwen3-0.6B)

| c | engine | out tok/s | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | ITL p50 ms | errors |
|---|---|---|---|---|---|---|---|
| 1 | tinyserve | 109 | 50.6 | 67.0 | 8.91 | 8.91 | 2 |
| 4 | tinyserve | 354 | 101.8 | 131.2 | 10.88 | 10.80 | 2 |
| 16 | tinyserve | 870 | 429.0 | 485.7 | 16.57 | 16.60 | 2 |
| 64 | tinyserve | 1,461 | 1231.7 | 1784.3 | 38.91 | 37.26 | 2 |
| 128 | tinyserve | 1,636 | 2197.0 | 3471.6 | 69.27 | 64.62 | 2 |
| 16 (repeat) | tinyserve | 846 | 436.6 | 525.1 | 17.04 | 16.89 | 2 |
| 1 | vllm | 161 | 33.9 | 44.3 | 6.07 | 6.07 | 2 |
| 4 | vllm | 544 | 61.7 | 79.0 | 7.09 | 7.07 | 2 |
| 16 | vllm | 1,352 | 120.2 | 289.9 | 11.26 | 10.93 | 2 |
| 64 | vllm | 2,136 | 237.8 | 1327.7 | 28.34 | 26.39 | 2 |
| 128 | vllm | 2,310 | 1375.8 | 2696.1 | 48.31 | 45.39 | 2 |
| 16 (repeat) | vllm | 1,349 | 116.3 | 294.0 | 11.28 | 10.96 | 2 |

| c | tinyserve / vLLM throughput |
|---|---|
| 1 | 68.0% |
| 4 | 65.0% |
| 16 | 64.3% |
| 64 | 68.4% |
| 128 | 70.8% |

## W2 (shared prefix, c=16)

| config | TTFT p50 ms | TTFT p99 ms | out tok/s |
|---|---|---|---|
| tinyserve cache off | 1051.6 | 1781.5 | 371 |
| tinyserve cache on | 197.6 | 239.7 | 826 |
| vLLM cache on | 160.1 | 196.0 | 1,135 |

## W3 GSM8K: DFlash speedup

| c | engine | base tok/s | DFlash tok/s | speedup | tau |
|---|---|---|---|---|---|
| 1 | tinyserve | 26.3 | 76.5 | 2.90x | 5.653 |
| 2 | tinyserve | 48.5 | 149.1 | 3.08x | 5.905 |
| 4 | tinyserve | 93.6 | 297.0 | 3.17x | 6.19 |
| 8 | tinyserve | 165.7 | 526.7 | 3.18x | 6.129 |
| 16 | tinyserve | 270.6 | 818.8 | 3.03x | 6.118 |
| 1 | vllm | 30.2 | 129.1 | 4.27x | 5.639 |
| 2 | vllm | 58.0 | 233.6 | 4.03x | 5.873 |
| 4 | vllm | 111.0 | 455.7 | 4.11x | 5.994 |
| 8 | vllm | 206.6 | 784.0 | 3.79x | 6.028 |
| 16 | vllm | 343.2 | 1,042.5 | 3.04x | 6.048 |

## DFlash tau per dataset

| source | gsm8k | humaneval | mtbench |
|---|---|---|---|
| tinyserve | 6.02 | 6.15 | 2.72 |
| z-lab reference (HF) | 5.54 | 5.96 | 2.05 |
| vLLM | 6.05 | 6.10 | 2.77 |
| DFlash paper | 6.53 | 6.64 | 4.35 |

## CUDA graphs (M5)

| batch | eager ms | graph ms | speedup |
|---|---|---|---|
| 1 | 39.08 | 7.75 | 5.04x |
| 2 | 40.12 | 8.09 | 4.96x |
| 4 | 39.75 | 8.18 | 4.86x |
| 8 | 41.22 | 8.60 | 4.79x |
| 16 | 40.84 | 8.89 | 4.59x |
| 32 | 40.95 | 10.18 | 4.02x |
| 64 | 40.69 | 12.62 | 3.23x |
| 128 | 41.31 | 17.37 | 2.38x |
```

## M9c (GPU) — 2026-10-07 03:33:08 UTC — git `f51b51d`

Command: `uv run modal run modal_app.py::gpu_tests --suite m9`
Source file: `results/verify/m9.log`
Note: Run at git of commit 'feat(spec): CUDA graphs for the DFlash verify step'. Server re-run: uv run modal run modal_app.py::bench --engine tinyserve --suite spec_dflash -> GSM8K c=1 104.3 tok/s (3.96x over non-spec 26.3).

```text
spec verify graphs captured for [1, 2, 4, 8, 16]
M9c bs=1: eager 83.9 tok/s -> graph 112.1 tok/s (1.336x)
M9c c=16: eager 874.7 tok/s -> graph 896.1 tok/s
M9c tau bs1 5.653 -> 5.653; c16 6.023 -> 6.024; outputs exact 8/8, exact-or-near-tie 8/8
M9c bs=1 DFlash(graph verify) 112.1 vs non-spec (graphs) 27.4 tok/s -> 4.10x
.
M9 PASS bs1_eager_tok_s=83.9 bs1_graph_tok_s=112.1 bs1_speedup_graph_vs_eager=1.336 c16_eager_tok_s=874.7 c16_graph_tok_s=896.1 tau_bs1_eager=5.653 tau_bs1_graph=5.653 tau_c16_eager=6.023 tau_c16_graph=6.024 outputs_bs1_exact=8/8 outputs_bs1_exact_or_near_tie=8/8 bs1_nonspec_graph_tok_s=27.4 bs1_dflash_graph_speedup_vs_nonspec=4.096

==================================== PASSES ====================================
=========================== short test summary info ============================
PASSED tests/gpu/test_m9_gpu.py::test_spec_verify_graphs
1 passed in 191.16s (0:03:11)
```

## DoD: Modal apps — 2026-10-07 03:35:29 UTC — git `24c9989`

Command: `uv run modal app list --json`  (exit code 0)

```text
[
  {
    "app_id": "ap-WoP1nHSUmOWO4QzpJ9VWKV",
    "description": "tinyserve",
    "state": "stopped",
    "tasks": "0",
    "created_at": "2026-10-06 20:27:20-07:00",
    "stopped_at": "2026-10-06 20:32:25-07:00"
  },
  {
    "app_id": "ap-ll2dRkd58VMLCLpQOhXa2K",
    "description": "tinyserve",
    "state": "stopped",
    "tasks": "0",
    "created_at": "2026-10-06 20:23:23-07:00",
    "stopped_at": "2026-10-06 20:27:01-07:00"
  },
  {
    "app_id": "ap-izQYODgaz3bj25AVRie260",
    "description": "tinyserve",
    "state": "stopped",
    "tasks": "0",
    "created_at": "2026-10-06 19:57:45-07:00",
    "stopped_at": "2026-10-06 19:59:38-07:00"
  },
  {
    "app_id": "ap-2hLNEhl9BMpAhCfwByAxv7",
    "description": "tinyserve",
    "state": "stopped",
    "tasks": "0",
    "created_at": "2026-10-06 19:57:20-07:00",
    "stopped_at": "2026-10-06 20:19:59-07:00"
  },
  {
    "app_id": "ap-WAM75f1hhJTLBuNWtrTaRV",
    "description": "tinyserve",
    "state": "stopped",
    "tasks": "0",
    "created_at": "2026-10-06 19:55:32-07:00",
    "stopped_at": "2026-10-06 20:14:05-07:00"
  },
  {
    "app_id": "ap-4R8d5KChwLTf5ZmHdtXFQm",
    "description": "tinyserve",
    "state": "stopped",
    "tasks": "0",
    "created_at": "2026-10-06 19:41:24-07:00",
    "stopped_at": "2026-10-06 19:57:06-07:00"
  },
  {
    "app_id": "ap-YmlMF4FXjUm5fCrwJP6JxH",
    "description": "tinyserve",
    "state": "stopped",
    "tasks": "0",
    "created_at": "2026-10-06 19:35:36-07:00",
    "stopped_at": "2026-10-06 19:54:45-07:00"
  },
  {
    "app_id": "ap-IO5kImA8r9sbAOt9Td2wc0",
    "description": "tinyserve",
    "state": "stopped",
    "tasks": "0",
    "created_at": "2026-10-06 19:35:36-07:00",
    "stopped_at": "2026-10-06 19:41:10-07:00"
  },
  {
    "app_id": "ap-yvhiwzlDfe912ZsOlieJ3Q",
    "description": "tinyserve",
    "state": "stopped",
    "tasks": "0",
    "created_at": "2026-10-06 19:29:40-07:00",
    "stopped_at": "2026-10-06 19:33:42-07:00"
  },
  {
    "app_id": "ap-rBW96BjXZSNwRCzVEsErx1",
    "description": "tinyserve",
    "state": "stopped",
    "tasks": "0",
    "created_at": "2026-10-06 19:24:36-07:00",
    "stopped_at": "2026-10-06 19:29:12-07:00"
  },
  {
    "app_id": "ap-nhE3hbjt64FQ8lLGwBGL0c",
    "description": "tinyserve",
    "state": "stopped",
    "tasks": "0",
    "created_at": "2026-10-06 19:19:55-07:00",
    "stopped_at": "2026-10-06 19:32:16-07:00"
  },
  {
    "app_id": "ap-Lw2e4kxII9cIDOdXySvXoX",
    "description": "tinyserve",
    "state": "stopped",
    "tasks": "0",
    "created_at": "2026-10-06 19:05:32-07:00",
    "stopped_at": "2026-10-06 19:09:17-07:00"
  },
  {
    "app_id": "ap-nGbP3bvkIf30UGvRZLzPBc",
    "description": "tinyserve",
    "state": "stopped",
    "tasks": "0",
    "created_at": "2026-10-06 19:00:27-07:00",
    "stopped_at": "2026-10-06 19:01:49-07:00"
  },
  {
    "app_id": "ap-BhsnCOSy0w6QjPhhHvMxvi",
    "description": "tinyserve",
    "state": "stopped",
    "tasks": "0",
    "created_at": "2026-10-06 18:57:56-07:00",
    "stopped_at": "2026-10-06 18:59:13-07:00"
  },
  {
    "app_id": "ap-TvKadfOouXDsFyvW4VJjXG",
    "description": "tinyserve",
    "state": "stopped",
    "tasks": "0",
    "created_at": "2026-10-06 18:49:18-07:00",
    "stopped_at": "2026-10-06 18:55:33-07:00"
  },
  {
    "app_id": "ap-VWaZTjwxFzg1TOz2UfQNPa",
    "description": "tinyserve",
    "state": "stopped",
    "tasks": "0",
    "created_at": "2026-10-06 18:40:36-07:00",
    "stopped_at": "2026-10-06 18:44:30-07:00"
  },
  {
    "app_id": "ap-1avocjq9UivLFhFJI8AnZW",
    "description": "tinyserve",
    "state": "stopped",
    "tasks": "0",
    "created_at": "2026-10-06 18:37:54-07:00",
    "stopped_at": "2026-10-06 18:40:16-07:00"
  },
  {
    "app_id": "ap-TnqzdDqTqE0OztqBFotp2c",
    "description": "tinyserve",
    "state": "stopped",
    "tasks": "0",
    "created_at": "2026-10-06 18:35:29-07:00",
    "stopped_at": "2026-10-06 18:36:59-07:00"
  }
]
```

## DoD: billing — 2026-10-07 03:35:37 UTC — git `24c9989`

Command: `uv run modal billing report --for "this month" --json && uv run modal billing report --for today --show-resources`  (exit code 0)

```text
[
  {
    "object_id": "ap-1aJZzABjDQmqrluUiMtwFG",
    "description": "tinyserve",
    "environment": "main",
    "interval_start": "2026-10-07T00:00:00",
    "cost": "0.01793253"
  },
  {
    "object_id": "ap-1avocjq9UivLFhFJI8AnZW",
    "description": "tinyserve",
    "environment": "main",
    "interval_start": "2026-10-07T00:00:00",
    "cost": "0.03923860"
  },
  {
    "object_id": "ap-2hLNEhl9BMpAhCfwByAxv7",
    "description": "tinyserve",
    "environment": "main",
    "interval_start": "2026-10-07T00:00:00",
    "cost": "0.41712797"
  },
  {
    "object_id": "ap-4R8d5KChwLTf5ZmHdtXFQm",
    "description": "tinyserve",
    "environment": "main",
    "interval_start": "2026-10-07T00:00:00",
    "cost": "0.28799764"
  },
  {
    "object_id": "ap-BhsnCOSy0w6QjPhhHvMxvi",
    "description": "tinyserve",
    "environment": "main",
    "interval_start": "2026-10-07T00:00:00",
    "cost": "0.01714652"
  },
  {
    "object_id": "ap-IO5kImA8r9sbAOt9Td2wc0",
    "description": "tinyserve",
    "environment": "main",
    "interval_start": "2026-10-07T00:00:00",
    "cost": "0.09788408"
  },
  {
    "object_id": "ap-Lw2e4kxII9cIDOdXySvXoX",
    "description": "tinyserve",
    "environment": "main",
    "interval_start": "2026-10-07T00:00:00",
    "cost": "0.01489590"
  },
  {
    "object_id": "ap-TnqzdDqTqE0OztqBFotp2c",
    "description": "tinyserve",
    "environment": "main",
    "interval_start": "2026-10-07T00:00:00",
    "cost": "0.02234409"
  },
  {
    "object_id": "ap-TvKadfOouXDsFyvW4VJjXG",
    "description": "tinyserve",
    "environment": "main",
    "interval_start": "2026-10-07T00:00:00",
    "cost": "0.10985803"
  },
  {
    "object_id": "ap-VWaZTjwxFzg1TOz2UfQNPa",
    "description": "tinyserve",
    "environment": "main",
    "interval_start": "2026-10-07T00:00:00",
    "cost": "0.04034261"
  },
  {
    "object_id": "ap-WAM75f1hhJTLBuNWtrTaRV",
    "description": "tinyserve",
    "environment": "main",
    "interval_start": "2026-10-07T00:00:00",
    "cost": "0.34013062"
  },
  {
    "object_id": "ap-YmlMF4FXjUm5fCrwJP6JxH",
    "description": "tinyserve",
    "environment": "main",
    "interval_start": "2026-10-07T00:00:00",
    "cost": "0.35129738"
  },
  {
    "object_id": "ap-izQYODgaz3bj25AVRie260",
    "description": "tinyserve",
    "environment": "main",
    "interval_start": "2026-10-07T00:00:00",
    "cost": "0.02999845"
  },
  {
    "object_id": "ap-ll2dRkd58VMLCLpQOhXa2K",
    "description": "tinyserve",
    "environment": "main",
    "interval_start": "2026-10-07T00:00:00",
    "cost": "0.04406740"
  },
  {
    "object_id": "ap-nGbP3bvkIf30UGvRZLzPBc",
    "description": "tinyserve",
    "environment": "main",
    "interval_start": "2026-10-07T00:00:00",
    "cost": "0.01924061"
  },
  {
    "object_id": "ap-nhE3hbjt64FQ8lLGwBGL0c",
    "description": "tinyserve",
    "environment": "main",
    "interval_start": "2026-10-07T00:00:00",
    "cost": "0.22384751"
  },
  {
    "object_id": "ap-rBW96BjXZSNwRCzVEsErx1",
    "description": "tinyserve",
    "environment": "main",
    "interval_start": "2026-10-07T00:00:00",
    "cost": "0.07069778"
  },
  {
    "object_id": "ap-xQP2lA4wh6GtUoEpuz9F0y",
    "description": "tinyserve",
    "environment": "main",
    "interval_start": "2026-10-07T00:00:00",
    "cost": "0.00620672"
  },
  {
    "object_id": "ap-yvhiwzlDfe912ZsOlieJ3Q",
    "description": "tinyserve",
    "environment": "main",
    "interval_start": "2026-10-07T00:00:00",
    "cost": "0.06829409"
  }
]
┌────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐
│            │            │            │ Interval   │            │            │
│ Object ID  │ Descripti… │ Environme… │ Start      │ Resource   │ Cost       │
├────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤
│ ap-1aJZzA… │ tinyserve  │ main       │ 2026-10-07 │ Network    │ 0.00973048 │
│            │            │            │            │ Egress     │            │
│ ap-1aJZzA… │ tinyserve  │ main       │ 2026-10-07 │ Memory     │ 0.00067815 │
│ ap-1aJZzA… │ tinyserve  │ main       │ 2026-10-07 │ CPU        │ 0.00752390 │
│ ap-1avocj… │ tinyserve  │ main       │ 2026-10-07 │ L4         │ 0.02809782 │
│ ap-1avocj… │ tinyserve  │ main       │ 2026-10-07 │ Memory     │ 0.00449565 │
│ ap-1avocj… │ tinyserve  │ main       │ 2026-10-07 │ CPU        │ 0.00664513 │
│ ap-2hLNEh… │ tinyserve  │ main       │ 2026-10-07 │ L4         │ 0.29866691 │
│ ap-2hLNEh… │ tinyserve  │ main       │ 2026-10-07 │ Network    │ 4.3E-7     │
│            │            │            │            │ Egress     │            │
│ ap-2hLNEh… │ tinyserve  │ main       │ 2026-10-07 │ Memory     │ 0.04778671 │
│ ap-2hLNEh… │ tinyserve  │ main       │ 2026-10-07 │ CPU        │ 0.07067393 │
│ ap-4R8d5K… │ tinyserve  │ main       │ 2026-10-07 │ L4         │ 0.20622204 │
│ ap-4R8d5K… │ tinyserve  │ main       │ 2026-10-07 │ Network    │ 2.4E-7     │
│            │            │            │            │ Egress     │            │
│ ap-4R8d5K… │ tinyserve  │ main       │ 2026-10-07 │ Memory     │ 0.03299553 │
│ ap-4R8d5K… │ tinyserve  │ main       │ 2026-10-07 │ CPU        │ 0.04877983 │
│ ap-BhsnCO… │ tinyserve  │ main       │ 2026-10-07 │ L4         │ 0.01227821 │
│ ap-BhsnCO… │ tinyserve  │ main       │ 2026-10-07 │ Memory     │ 0.00196451 │
│ ap-BhsnCO… │ tinyserve  │ main       │ 2026-10-07 │ CPU        │ 0.00290380 │
│ ap-IO5kIm… │ tinyserve  │ main       │ 2026-10-07 │ L4         │ 0.07008457 │
│ ap-IO5kIm… │ tinyserve  │ main       │ 2026-10-07 │ Network    │ 1.8E-7     │
│            │            │            │            │ Egress     │            │
│ ap-IO5kIm… │ tinyserve  │ main       │ 2026-10-07 │ Memory     │ 0.01121353 │
│ ap-IO5kIm… │ tinyserve  │ main       │ 2026-10-07 │ CPU        │ 0.01658579 │
│ ap-Lw2e4k… │ tinyserve  │ main       │ 2026-10-07 │ L4         │ 0.01066659 │
│ ap-Lw2e4k… │ tinyserve  │ main       │ 2026-10-07 │ Memory     │ 0.00170665 │
│ ap-Lw2e4k… │ tinyserve  │ main       │ 2026-10-07 │ CPU        │ 0.00252265 │
│ ap-TnqzdD… │ tinyserve  │ main       │ 2026-10-07 │ L4         │ 0.01600007 │
│ ap-TnqzdD… │ tinyserve  │ main       │ 2026-10-07 │ Memory     │ 0.00256001 │
│ ap-TnqzdD… │ tinyserve  │ main       │ 2026-10-07 │ CPU        │ 0.00378402 │
│ ap-TvKadf… │ tinyserve  │ main       │ 2026-10-07 │ L4         │ 0.07866669 │
│ ap-TvKadf… │ tinyserve  │ main       │ 2026-10-07 │ Memory     │ 0.01258667 │
│ ap-TvKadf… │ tinyserve  │ main       │ 2026-10-07 │ CPU        │ 0.01860467 │
│ ap-VWaZTj… │ tinyserve  │ main       │ 2026-10-07 │ L4         │ 0.02888837 │
│ ap-VWaZTj… │ tinyserve  │ main       │ 2026-10-07 │ Memory     │ 0.00462214 │
│ ap-VWaZTj… │ tinyserve  │ main       │ 2026-10-07 │ CPU        │ 0.00683210 │
│ ap-WAM75f… │ tinyserve  │ main       │ 2026-10-07 │ L4         │ 0.24355934 │
│ ap-WAM75f… │ tinyserve  │ main       │ 2026-10-07 │ Memory     │ 0.03896949 │
│ ap-WAM75f… │ tinyserve  │ main       │ 2026-10-07 │ CPU        │ 0.05760178 │
│ ap-YmlMF4… │ tinyserve  │ main       │ 2026-10-07 │ L4         │ 0.25155559 │
│ ap-YmlMF4… │ tinyserve  │ main       │ 2026-10-07 │ Memory     │ 0.04024889 │
│ ap-YmlMF4… │ tinyserve  │ main       │ 2026-10-07 │ CPU        │ 0.05949290 │
│ ap-izQYOD… │ tinyserve  │ main       │ 2026-10-07 │ L4         │ 0.02148117 │
│ ap-izQYOD… │ tinyserve  │ main       │ 2026-10-07 │ Memory     │ 0.00343699 │
│ ap-izQYOD… │ tinyserve  │ main       │ 2026-10-07 │ CPU        │ 0.00508030 │
│ ap-ll2dRk… │ tinyserve  │ main       │ 2026-10-07 │ L4         │ 0.03155560 │
│ ap-ll2dRk… │ tinyserve  │ main       │ 2026-10-07 │ Memory     │ 0.00504890 │
│ ap-ll2dRk… │ tinyserve  │ main       │ 2026-10-07 │ CPU        │ 0.00746290 │
│ ap-nGbP3b… │ tinyserve  │ main       │ 2026-10-07 │ L4         │ 0.01377774 │
│ ap-nGbP3b… │ tinyserve  │ main       │ 2026-10-07 │ Memory     │ 0.00220444 │
│ ap-nGbP3b… │ tinyserve  │ main       │ 2026-10-07 │ CPU        │ 0.00325843 │
│ ap-nhE3hb… │ tinyserve  │ main       │ 2026-10-07 │ L4         │ 0.16029181 │
│ ap-nhE3hb… │ tinyserve  │ main       │ 2026-10-07 │ Memory     │ 0.02564669 │
│ ap-nhE3hb… │ tinyserve  │ main       │ 2026-10-07 │ CPU        │ 0.03790901 │
│ ap-rBW96B… │ tinyserve  │ main       │ 2026-10-07 │ L4         │ 0.04456487 │
│ ap-rBW96B… │ tinyserve  │ main       │ 2026-10-07 │ Network    │ 0.00057479 │
│            │            │            │            │ Egress     │            │
│ ap-rBW96B… │ tinyserve  │ main       │ 2026-10-07 │ Memory     │ 0.00726319 │
│ ap-rBW96B… │ tinyserve  │ main       │ 2026-10-07 │ CPU        │ 0.01829493 │
│ ap-xQP2lA… │ tinyserve  │ main       │ 2026-10-07 │ L4         │ 0.00444448 │
│ ap-xQP2lA… │ tinyserve  │ main       │ 2026-10-07 │ Memory     │ 0.00071112 │
│ ap-xQP2lA… │ tinyserve  │ main       │ 2026-10-07 │ CPU        │ 0.00105112 │
│ ap-yvhiwz… │ tinyserve  │ main       │ 2026-10-07 │ L4         │ 0.04888862 │
│ ap-yvhiwz… │ tinyserve  │ main       │ 2026-10-07 │ Network    │ 1.8E-7     │
│            │            │            │            │ Egress     │            │
│ ap-yvhiwz… │ tinyserve  │ main       │ 2026-10-07 │ Memory     │ 0.00782218 │
│ ap-yvhiwz… │ tinyserve  │ main       │ 2026-10-07 │ CPU        │ 0.01158311 │
└────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘
```

## DoD: clean clone — 2026-10-07 03:39:39 UTC — git `24c9989`

Command: `rm -rf ../tinyserve-verify && git clone -q . ../tinyserve-verify && cd ../tinyserve-verify && uv sync -q && uv run pytest tests/cpu -q -m "not slow"`  (exit code 0)

```text
........................................................................ [ 91%]
.......                                                                  [100%]
============================== warnings summary ===============================
.venv\Lib\site-packages\fastapi\testclient.py:1
  C:\Users\tranb\projects\tinyserve-verify\.venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
79 passed, 1 deselected, 1 warning in 81.14s (0:01:21)
warning: `VIRTUAL_ENV=C:\Users\tranb\projects\tinyserve\.venv` does not match the project environment path `.venv` and will be ignored; use `--active` to target the active environment instead
```
