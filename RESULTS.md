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
