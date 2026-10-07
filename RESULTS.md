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
