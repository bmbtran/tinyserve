# tinyserve: resources

Everything here was checked on 2026-10-06. Each entry has one line on why it matters. Repo file paths can move between versions; if a path 404s, search the repo for the class/function name given.

## 1. Pinned versions (source of truth; PLAN.md §3 explains the fallbacks)

| Thing | Pin | Where it was verified |
|---|---|---|
| Local Python | 3.12 via `uv` (system has 3.14; torch 2.8 has no cp314 wheels) | https://docs.astral.sh/uv/ |
| torch (engine image + local) | 2.8.0 (Linux default wheel = CUDA 12.8, bundles triton 3.4.0) | PyPI `torch/2.8.0` metadata |
| flash-attn | 2.8.3.post1 prebuilt wheel `flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl` (ABI FALSE variant and a torch 2.9 wheel exist as fallbacks) | https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3.post1 |
| transformers | 4.57.3 (the DFlash authors evaluated with it) | https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16 |
| modal client | 1.6.1 | PyPI `modal` |
| vLLM baseline | 0.30.0 (torch 2.13.0, CUDA 13.0.3, flashinfer 0.6.18.post1). Fallback: 0.21.0 on `nvidia/cuda:12.9.0-devel-ubuntu22.04` (Modal's tested example stack). Avoid 0.31.0 (released 2026-10-05). | PyPI `vllm` |
| Modal host driver | 580.95.05 / CUDA driver API 13.0 | https://modal.com/docs/guide/cuda |
| GPU | Modal `L4` (sm_89, 24 GB). Not T4: FA2 does not support Turing. | https://modal.com/docs/guide/gpu |

## 2. Papers

### Headline feature
- **DFlash: Block Diffusion for Flash Speculative Decoding**, Chen, Liang, Liu (2026). https://arxiv.org/abs/2602.06036 (HTML: https://arxiv.org/html/2602.06036v1). The headline technique: a 5-layer drafter predicts a 16-token block in one non-causal pass, conditioned on target hidden states injected into the draft KV. Table 1 gives the Qwen3-4B τ we compare against: GSM8K 6.53, MATH-500 7.84, HumanEval 6.64, MBPP 6.09, MT-Bench 4.35. Table 3 shows the speedup falling from 4.8× at c = 1 to 2.9× at c = 32 (SGLang, B200).
- **Block Diffusion (BD3-LM)**, Arriola et al. (ICLR 2025). https://arxiv.org/abs/2503.09573. Background on block diffusion LMs, the idea DFlash borrows for drafting.

### Speculative decoding lineage
- **Fast Inference from Transformers via Speculative Decoding**, Leviathan et al. (2022). https://arxiv.org/abs/2211.17192. The original draft/verify algorithm and its lossless guarantee.
- **Accelerating LLM Decoding with Speculative Sampling**, Chen et al. (2023). https://arxiv.org/abs/2302.01318. Rejection sampling for temperature > 0 (concept for LEARNING.md; out of scope to implement).
- **Medusa** (2024). https://arxiv.org/abs/2401.10774. Multiple decoding heads; useful contrast with feature-level drafting.
- **EAGLE** (2024). https://arxiv.org/abs/2401.15077, **EAGLE-2** https://arxiv.org/abs/2406.16858, **EAGLE-3** (NeurIPS 2025) https://arxiv.org/abs/2503.01840. The state-of-the-art autoregressive feature drafter, our fallback headline, and the main baseline in the DFlash paper.
- **SuffixDecoding** (NeurIPS 2025). https://arxiv.org/abs/2411.04975. A model-free suffix-tree drafter for agentic workloads; candidate stretch proposer.

### Engine foundations
- **PagedAttention / vLLM**, Kwon et al. (SOSP 2023). https://arxiv.org/abs/2309.06180. Block-based KV cache and block tables: the core memory design.
- **Orca**, Yu et al. (OSDI 2022). https://www.usenix.org/conference/osdi22/presentation/yu. Iteration-level scheduling, i.e. continuous batching.
- **SGLang / RadixAttention**, Zheng et al. (2024). https://arxiv.org/abs/2312.07104. Radix-tree prefix caching; we implement the simpler hash-chain variant and explain the difference.
- **Sarathi-Serve (chunked prefill)**, Agrawal et al. (OSDI 2024). https://arxiv.org/abs/2403.02310. Chunked prefill and stall-free batching (stretch M9a).
- **FlashAttention**, https://arxiv.org/abs/2205.14135, and **FlashAttention-2**, https://arxiv.org/abs/2307.08691. The IO-aware attention kernels we call through flash-attn.
- **FlashInfer** (2025). https://arxiv.org/abs/2501.01005. An alternative attention-kernel library (used inside vLLM); cited as an alternative backend.
- **Qwen3 Technical Report** (2025). https://arxiv.org/abs/2505.09388. Model architecture details (QK-norm, GQA, thinking mode).

### Related work (cite, don't build)
- **DistServe** (OSDI 2024). https://arxiv.org/abs/2401.09670. Disaggregated prefill/decode; rejected as headline (needs ≥ 2 GPUs).
- **Mooncake** (FAST 2025). https://arxiv.org/abs/2407.00079. KV-centric disaggregated serving at Moonshot; context for interviews.
- **NebulaSD** (2026). https://arxiv.org/abs/2609.29364 and **StreamServe** (2026) https://arxiv.org/abs/2604.09562. Recent 2026 serving/spec papers found during the survey; related work only.

## 3. Reference code (what to read, in this order)

### nano-vllm (~1.2k lines; closest design reference; MIT): https://github.com/GeeeekExplorer/nano-vllm
- `nanovllm/engine/block_manager.py`: xxhash chained prefix hashing, ref counts, `can_allocate`/`may_append`.
- `nanovllm/engine/scheduler.py`: prefill-first scheduling, preemption by recompute.
- `nanovllm/engine/sequence.py`: per-request state.
- `nanovllm/engine/model_runner.py`: KV pool sizing from free memory, `prepare_prefill`/`prepare_decode` (slot_mapping, cu_seqlens, block_tables), CUDA graph capture/replay with padding.
- `nanovllm/layers/attention.py`: the exact flash-attn call pattern (`flash_attn_varlen_func(..., block_table=...)` for prefill/prefix, `flash_attn_with_kvcache` for decode) and the Triton `store_kvcache_kernel` (our one allowed Triton kernel is modelled on it).
- `nanovllm/models/qwen3.py`, `nanovllm/utils/loader.py`, `nanovllm/utils/context.py`: Qwen3 layers, weight loading, the global attention context.
- `bench.py`: their throughput benchmark (they report 1434 tok/s vs vLLM 1362 on an RTX 4070 laptop, Qwen3-0.6B). Use it to sanity-check our order of magnitude.
- Differentiation: nano-vllm has no speculative decoding, no HTTP server, no latency metrics, no CPU test backend, and no same-GPU vLLM latency comparison.

### mini-sglang (~5k lines): https://github.com/sgl-project/mini-sglang (blog: https://lmsys.org/blog/2025-12-17-minisgl)
- Read its radix cache, scheduler, and OpenAI server modules for a second, more production-like design (radix attention, chunked prefill, overlap scheduling).

### vLLM: https://github.com/vllm-project/vllm
- `vllm/v1/core/block_pool.py` and `vllm/v1/core/kv_cache_manager.py`: production hash-based prefix caching (APC) with lookahead slots.
- `vllm/v1/core/sched/scheduler.py`: the unified token-budget scheduler.
- `vllm/v1/spec_decode/`: how vLLM integrates drafters (EAGLE/EAGLE-3/DFlash proposers, rejection sampler).
- `vllm/model_executor/models/llama_eagle3.py`: EAGLE-3 draft layout (needed only if falling back to EAGLE-3).
- `vllm/model_executor/models/qwen3.py`: production Qwen3.
- `vllm/benchmarks/serve.py`: metric definitions (TTFT/TPOT/ITL) we mirror in `bench/client.py`.
- Docs: https://docs.vllm.ai/ (speculative decoding page; `vllm serve` CLI flags).

### SGLang: https://github.com/sgl-project/sglang
- `python/sglang/srt/mem_cache/radix_cache.py`: RadixAttention implementation (for LEARNING.md).
- `python/sglang/srt/speculative/`: EAGLE/DFlash workers (DFlash integration PR referenced from the z-lab repo).

### DFlash reference: https://github.com/z-lab/dflash and https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16
- `dflash.py` / `utils.py` in the HF repo (MIT): `DFlashDraftModel.forward` and `spec_generate`. This is the ~60-line ground-truth loop that `tests/reference/dflash_ref.py` vendors. Note: `extract_context_feature` uses `hidden_states[layer_id + 1]`, `crop(start)` on the draft cache, and τ counts the bonus token.
- `config.json`: `block_size=16`, `mask_token_id=151669`, `target_layer_ids=[1,9,17,25,33]`, 5 layers.
- The README covers launching with SGLang/vLLM (`--speculative-config '{"method":"dflash",...,"num_speculative_tokens":15}'`) and notes the draft was trained with thinking disabled.

### EAGLE-3 fallback heads (only if DFlash is abandoned)
- https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3 (99k downloads; `pytorch_model.bin`, 274 MB) and https://huggingface.co/AngelSlim/Qwen3-4B_eagle3 (safetensors). Keys: `fc`, `midlayer.*` (q/k/v take 2×hidden input), `lm_head [32000,H]`, `d2t`, `t2d`. The README reports Qwen3-1.7B τ≈2.17, 1.7× (vLLM 0.11.2, H20, k = 2).
- Official EAGLE repo: https://github.com/SafeAILab/EAGLE.

### flash-attention: https://github.com/Dao-AILab/flash-attention
- `README.md`, section on `flash_attn_with_kvcache`: the paged-KV `block_table` argument; **page block size must be a multiple of 256**.
- `flash_attn/flash_attn_interface.py`: `flash_attn_varlen_func` signature, including `block_table` (used for prefix-cached prefill, spec verify, and the DFlash draft with `causal=False`).

### Triton
- Tutorial 01, vector add: https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html. The only Triton concepts needed (program_id, arange, load/store, masks) for the KV-store kernel.

## 4. Modal docs
- Pricing: https://modal.com/pricing. Per-second rates (L4 $0.000222/s ≈ $0.80/h; A10 $1.10/h; L40S $1.95/h; H100 $3.95/h; CPU $0.0000131/core/s; memory $0.00000222/GiB/s; Starter plan $30/month free credits; Volumes 1 TiB free).
- GPU guide: https://modal.com/docs/guide/gpu. Valid `gpu=` strings (`"L4"`, `"A10"`, `"L40S"`, `"H100"` …) and fallback-list semantics (we deliberately don't use fallbacks).
- CUDA guide: https://modal.com/docs/guide/cuda. Host driver 580.95.05 / CUDA 13.0; pip torch needs no toolkit image.
- Install flash-attn example: https://modal.com/docs/examples/install_flash_attn. The prebuilt-wheel + `uv_pip_install` pattern we copy (their example: FA 2.7.4.post1 / torch 2.6 / py3.13).
- vLLM inference example: https://modal.com/docs/examples/vllm_inference. The vLLM image pattern (cuda 12.9 devel + py3.12 + vLLM 0.21.0) and HF/vLLM cache volumes; our fallback baseline stack.
- Images: https://modal.com/docs/guide/images. `debian_slim`, `uv_pip_install`, `env`, `add_local_python_source` / `add_local_dir` (runtime mounts → no rebuild on code edits).
- Volumes: https://modal.com/docs/guide/volumes. Weight caching; `volume.commit()` after downloads.
- Timeouts: https://modal.com/docs/guide/timeouts. Default 300 s, max 24 h; `startup_timeout` separate.
- Scaling / scaledown: https://modal.com/docs/guide/scale. `scaledown_window`, `max_containers`, `min_containers` (never set).
- Billing CLI: https://modal.com/docs/reference/cli/billing. `modal billing report --for "this month" --json` / `--show-resources`; used by `scripts/cost_guard.py` and COSTLOG.md.
- Apps / ephemeral runs: https://modal.com/docs/guide/apps. `modal run` = ephemeral app that stops when the entrypoint returns; `modal app list` / `modal app stop`.

## 5. Datasets (all ungated)
- GSM8K: https://huggingface.co/datasets/openai/gsm8k (MIT). W3 spec prompts (DFlash paper's headline dataset).
- HumanEval: https://huggingface.co/datasets/openai/openai_humaneval (MIT). W3 code prompts.
- MT-Bench prompts: https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts. W3 chat prompts (lowest DFlash τ, a good stress case).
- ShareGPT (optional W4): https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered. Realistic length distribution, the de-facto serving benchmark.

## 6. Models (all ungated)
- https://huggingface.co/Qwen/Qwen3-0.6B: dev target (28 layers, hidden 1024, 16/8 heads, head_dim 128, tied embeddings).
- https://huggingface.co/Qwen/Qwen3-4B: spec target (36 layers, hidden 2560, 32/8 heads, head_dim 128, tied embeddings).
- https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16: DFlash draft (MIT, 1.07 GB, created 2026-01-04).
- https://huggingface.co/Qwen/Qwen3-1.7B: optional extra benchmark point.

## 7. Blogs / books / context
- Baseten, *Inference Engineering* (book): https://www.baseten.co/inference-engineering/book. The owner's study text; chapters on hardware, software and techniques map to LEARNING.md sections.
- vLLM blog, Speculators v0.5 (DFlash support, 2026-05-28): https://vllm.ai/blog/2026-05-28-speculators-v050. Evidence that DFlash is production-relevant in 2026.
- DFlash project page: https://z-lab.ai/projects/dflash/. Figures and a summary of results.
- Google Developers blog on diffusion-style speculative decoding on TPUs: https://developers.googleblog.com/supercharging-llm-inference-on-google-tpus-achieving-3x-speedups-with-diffusion-style-speculative-decoding/. Industry adoption signal for the README's "why this matters".
- LMSYS mini-sglang blog: https://lmsys.org/blog/2025-12-17-minisgl. How a minimal engine is structured; useful for the README comparison.
- uv docs: https://docs.astral.sh/uv/. `uv python pin`, `uv sync`, `uv run`.
