# COSTLOG

One row per GPU run (written automatically by `modal_app.py`). est $ = wall minutes x $1.12/60 (L4 + 4 CPU + 16 GiB).

| date (UTC) | command | GPU | wall min | est $ (wall×$1.12/60) | billing month-to-date $ | purpose / outcome |
|---|---|---|---|---|---|---|
| 2026-10-07 01:27 | `modal run modal_app.py::smoke` | L4 | 0.3 | 0.006 | 0.03 | M0 env smoke |
| 2026-10-07 01:36 | `modal run modal_app.py::gpu_tests --suite m2` | L4 | 1.2 | 0.022 | 0.06 | M2 GPU tests: FAIL |
| 2026-10-07 01:40 | `modal run modal_app.py::gpu_tests --suite m2` | L4 | 2.0 | 0.038 | 0.14 | M2 GPU tests: FAIL |
| 2026-10-07 01:44 | `modal run modal_app.py::gpu_tests --suite m2` | L4 | 3.6 | 0.066 | 0.22 | M2 GPU tests: FAIL |
| 2026-10-07 01:55 | `modal run modal_app.py::gpu_tests --suite m3` | L4 | 5.9 | 0.111 | 0.42 | M3 GPU tests: FAIL |
| 2026-10-07 01:59 | `modal run modal_app.py::gpu_tests --suite m4` | L4 | 1.0 | 0.018 | 0.47 | M4 GPU tests: FAIL |
| 2026-10-07 02:01 | `modal run modal_app.py::gpu_tests --suite m5` | L4 | 1.0 | 0.019 | 0.51 | M5 GPU tests: PASS |
| 2026-10-07 02:09 | `modal run modal_app.py::gpu_tests --suite m6` | L4 | 3.4 | 0.064 | 0.54 | M6 GPU tests: PASS |
| 2026-10-07 02:29 | `modal run modal_app.py::vllm_smoke` | L4 | 3.3 | 0.061 | 0.96 | M8a vLLM smoke: ERROR |
| 2026-10-07 02:32 | `modal run modal_app.py::gpu_tests --suite m7` | L4 | 12.0 | 0.224 | 1.15 | M7 GPU tests: FAIL |
| 2026-10-07 02:33 | `modal run modal_app.py::vllm_smoke` | L4 | 3.7 | 0.069 | 1.23 | M8a vLLM smoke: ok |
| 2026-10-07 02:41 | `modal run modal_app.py::vllm_smoke --dflash` | L4 | 5.2 | 0.097 | 1.61 | M8a vLLM smoke: ok |
| 2026-10-07 02:54 | `modal run modal_app.py::bench --engine tinyserve --suite core` | L4 | 18.8 | 0.351 | 2.48 | M8 bench tinyserve/core: ok |
| 2026-10-07 02:57 | `modal run modal_app.py::bench --engine vllm --suite core` | L4 | 15.4 | 0.288 | 2.74 | M8 bench vllm/core: ok |
| 2026-10-07 02:59 | `modal run modal_app.py::gpu_tests --suite profile` | L4 | 1.5 | 0.028 | 2.87 | PROFILE GPU tests: PASS |
| 2026-10-07 03:14 | `modal run modal_app.py::bench --engine tinyserve --suite spec` | L4 | 18.3 | 0.341 | 3.98 | M8 bench tinyserve/spec: ok |
| 2026-10-07 03:19 | `modal run modal_app.py::bench --engine vllm --suite spec` | L4 | 22.3 | 0.417 | 4.31 | M8 bench vllm/spec: ok |
| 2026-10-07 03:27 | `modal run modal_app.py::gpu_tests --suite m9` | L4 | 3.3 | 0.062 | 4.44 | M9 GPU tests: PASS |
| 2026-10-07 03:32 | `modal run modal_app.py::bench --engine tinyserve --suite spec_dflash` | L4 | 4.8 | 0.089 | 4.44 | M8 bench tinyserve/spec_dflash: ok |
| 2026-10-07 04:01 | `modal run modal_app.py::gpu_tests --suite m5` | L4 | 1.1 | 0.021 | 4.65 | M5 GPU tests: PASS |
| 2026-10-07 04:05 | `modal run modal_app.py::gpu_tests --suite mtbench` | L4 | 2.3 | 0.043 | 4.92 | MTBENCH GPU tests: PASS |
| 2026-10-07 04:07 | `modal run modal_app.py::gpu_tests --suite lossless` | L4 | 4.9 | 0.092 | 5.30 | LOSSLESS GPU tests: PASS |
| 2026-10-07 04:14 | `modal run modal_app.py::gpu_tests --suite fp32` | L4 | 11.6 | 0.216 | 5.86 | FP32 GPU tests: PASS |
| 2026-10-07 04:35 | `modal run modal_app.py::gpu_tests --suite m6` | L4 | 34.6 | 0.647 | 6.62 | M6 GPU tests: ERROR |
