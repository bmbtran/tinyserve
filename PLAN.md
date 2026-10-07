# tinyserve: implementation spec (one-shot build plan)

> **Who this is for:** a fresh Claude Code session that will build the whole project using only this file and `RESOURCES.md`.
> **Planning date:** 2026-10-06. Every version pin, price, and model ID below was checked on that date. Re-check anything marked **[VERIFY]** before you rely on it.
> **Owner:** a student building a portfolio (target companies: Nebius, Modal, Cognition, Decagon, Kalshi, Polymarket). The owner has never written PyTorch internals, CUDA, or Triton. Code must be readable, well commented, and must use library kernels. It may contain at most one small Triton kernel.

---

## 0. TL;DR

Build **tinyserve**, a small (~2.5–3.5k lines of Python) LLM inference engine for **Qwen3** with:

1. **A paged KV cache.** A block manager in the style of PagedAttention (vLLM).
2. **Continuous batching.** Iteration-level scheduling in the style of Orca, with preemption by recompute.
3. **Prefix caching.** Hash-based, at block granularity, the same idea as vLLM's automatic prefix caching (APC). RadixAttention is explained in the write-up.
4. **CUDA graphs** for the non-speculative decode path.
5. **An OpenAI-compatible streaming HTTP server** (FastAPI + SSE).
6. **Headline feature: DFlash speculative decoding** (block-diffusion drafter; Chen, Liang, Liu, arXiv **2602.06036**, Feb 2026). It uses the pretrained, MIT-licensed draft `z-lab/Qwen3-4B-DFlash-b16` for the target `Qwen/Qwen3-4B`. It is lossless (greedy output must match non-speculative output) and needs no training. The draft is integrated with the paged KV cache and continuous batching.

All GPU work runs on **Modal** with **L4** GPUs, about $0.80/hr for the GPU alone. Planned spend is about **$16**, with a hard stop at **$25** and a $30/month cap.

Benchmarks run **against vLLM on the same GPU, model and workload**. The output is JSON plus matplotlib charts committed to the repo, a blog-style README, and a `LEARNING.md` for interview prep.

---

## 1. Goals and non-goals

### Goals
- G1. Correct greedy generation for Qwen3-0.6B / 1.7B / 4B. It must be checked against HF `transformers`, exactly in fp32/fp64 on CPU and within a near-tie rule in bf16 on GPU.
- G2. Paged KV cache plus continuous batching with preemption. Pure-Python scheduling logic, unit-tested on CPU.
- G3. Hash-based prefix caching with a measurable hit rate and TTFT win.
- G4. CUDA-graph decode with a measurable ITL win.
- G5. OpenAI-compatible `/v1/completions` and `/v1/chat/completions`, with and without streaming.
- G6. DFlash speculative decoding: lossless, batched, on the paged KV cache. Its acceptance length τ must be reported next to the paper's values and the reference implementation's values.
- G7. A reproducible benchmark harness: tinyserve vs vLLM on the same L4, with TTFT, ITL/TPOT, throughput vs concurrency, and spec-decode speedup vs concurrency.
- G8. An honest write-up that includes "what didn't work / didn't reproduce".

### Non-goals (do NOT build these; mention them in the README as future work)
- Tensor or pipeline parallelism, multi-GPU, or disaggregated prefill/decode.
- Quantization (FP8/INT4), LoRA, multimodal input, MoE models, or sliding-window models.
- Writing attention kernels. Use `flash-attn` 2.8.3. The only allowed custom kernel is the optional Triton KV-store kernel.
- `torch.compile`. Compile time burns paid GPU minutes and adds failure modes.
- Lossless speculative *sampling* at temperature > 0. Spec decode is greedy-only. Non-spec decoding supports temperature/top-p/top-k.
- CUDA graphs for the spec-decode path (stretch only).
- Overlap scheduling, structured output, logprobs, or stop *strings*. Stop *token IDs* are supported.
- Chunked prefill (stretch milestone M9 only).

---

## 2. Why DFlash: ranked shortlist of candidate headline papers

Selection rules:
- (1) works on small models,
- (2) has verifiable numbers or code,
- (3) fits the $30 Modal budget,
- (4) is relevant to how Nebius, Modal and Baseten serve models,
- (+) lossless is preferred because it gives a hard correctness test.

| Rank | Candidate | Year | (1) small models | (2) numbers/code | (3) budget | (4) relevance | Verdict |
|---|---|---|---|---|---|---|---|
| **1** | **DFlash: block-diffusion drafter** (arXiv 2602.06036) | 2026 | Pretrained drafts for **Qwen3-4B** and **8B** (`z-lab/Qwen3-4B-DFlash-b16`, MIT, ungated, 1.07 GB). No 0.6B/1.7B drafts. | Paper Table 1: Qwen3-4B τ = 6.53 (GSM8K), 6.64 (HumanEval), 4.35 (MT-Bench); 5.15× speedup at bs = 1 on H200. A ~60-line reference `spec_generate` ships in the HF repo. Integrated into vLLM (≥ ~0.25) and SGLang. | 4B bf16 (8 GB) + draft (1 GB) fits a 24 GB L4. | Diffusion-style drafting is the 2026 trend (vLLM Speculators v0.5, SGLang, Google TPU blog). | **CHOSEN** |
| 2 | **EAGLE-3** (arXiv 2503.01840, NeurIPS'25) | 2025 | `AngelSlim/Qwen3-1.7B_eagle3` and `AngelSlim/Qwen3-4B_eagle3` exist, are ungated, and have 1-layer heads. | AngelSlim reports Qwen3-1.7B τ ≈ 2.17, 1.7× on vLLM 0.11.2 (H20, k = 2). DFlash paper reports EAGLE-3 τ ≈ 3.3 on Qwen3-4B. | Cheap. | In every production engine. | **Fallback** if DFlash fails. Needs *sequential* draft steps with per-step draft KV bookkeeping, so it is more complex in a paged engine and gives a smaller speedup. |
| 3 | SuffixDecoding (arXiv 2411.04975, NeurIPS'25) | 2024/25 | Model-free, works on any model. | Up to 5.3× on agentic workloads. Code in ArcticInference. | Very cheap. | Snowflake/vLLM. | It only wins on repetitive/agentic traffic. Good **stretch**: it plugs into the same verify path (the `Proposer` interface). |
| 4 | PARD (`amd/PARD-Qwen3-0.6B`) | 2025 | A target-independent parallel draft. | Some numbers. | Cheap. | Lower. | Less adopted; skip. |
| 5 | Disaggregated prefill/decode (DistServe 2401.09670 / Mooncake 2407.00079) | 2024/25 | — | — | Needs ≥ 2 GPUs, which doubles cost. Wins only appear at scale under SLOs. | High. | **Rejected.** A toy 2-GPU KV handoff in one container would not reproduce the papers' gains on the budget. |
| 6 | Others found (NebulaSD 2609.29364, StreamServe 2604.09562, KV-compression papers) | 2026 | Mostly no small pretrained artifacts. | Mixed. | — | — | Skip; cite as related work. |

**Why DFlash wins:**
- It is the newest lossless technique that has (a) pretrained Qwen3 drafts, (b) a tiny, readable reference implementation, (c) published per-dataset τ for the exact target model, and (d) a production integration in vLLM. That last point gives an apples-to-apples baseline: **vLLM+DFlash vs tinyserve+DFlash on the same L4**.
- It fits a paged engine unusually well. The draft runs **one** non-causal forward per step (no sequential draft loop). Verification is simply a 16-token "mini-prefill" through the same `flash_attn_varlen_func(block_table=...)` path that prefix-cached prefill already uses.
- Acceptance length τ is hardware-independent: it depends only on the models and prompts. So τ can be checked against the paper even on a cheap L4.

**Honest caveats** (repeat these in the README):
- DFlash drafts exist only for Qwen3-4B and up, so the spec-decode experiments use **Qwen3-4B**. Core engine development uses Qwen3-0.6B, which is cheap and fast.
- Speedups shrink as concurrency grows: each sequence verifies 16 tokens per step, so large batches become compute-bound. The paper's SGLang table already shows 4.8× at c = 1 falling to 2.9× at c = 32 on B200. The L4 has much less compute, so the crossover should come earlier. **Chart this.** It is an interesting result whichever way it comes out.
- The drafts were trained with **thinking mode disabled**. Always use `enable_thinking=False` in the chat template for spec benchmarks.

---

## 3. Pinned environment (verified 2026-10-06)

### 3.1 Local (Windows 11 laptop, no NVIDIA GPU)
- Python **3.12** via `uv`. The system Python is 3.14, which is unsupported: torch 2.8 has no cp314 wheels.
  - If `uv` is missing, install it: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`.
  - `uv python pin 3.12`, which writes `.python-version`.
- `pyproject.toml` requires `requires-python = ">=3.12,<3.13"`.
- Local deps (CPU-only; PyPI torch on Windows is CPU-only):
  - `torch==2.8.0`, `transformers==4.57.3`, `safetensors`, `xxhash`, `numpy`, `fastapi`, `uvicorn`, `httpx`, `modal==1.6.1`, `huggingface_hub`
  - dev: `pytest`, `matplotlib`, `datasets`, `pandas`, `pyarrow`
- **Do not** list `flash-attn` or `triton` as local deps. Import them lazily, guarded by `try/except ImportError`.
- Modal token: already configured by the user (`~/.modal.toml`). Check with `uv run modal profile current`.
- **Important:** `C:\Users\tranb` is itself a git repository. Run `git init` **inside** `C:\Users\tranb\projects\tinyserve` (milestone M0) so commits never land in the home repo. Check with `git rev-parse --show-toplevel`, which must print `.../projects/tinyserve`.

### 3.2 Modal engine image (tinyserve)

| Component | Pin | Notes |
|---|---|---|
| Base | `modal.Image.debian_slim(python_version="3.12")` | Host driver 580.95.05 / CUDA 13.0 per Modal docs. PyPI torch bundles its own CUDA runtime, so no CUDA toolkit image is needed. |
| torch | `2.8.0` (default PyPI wheel = CUDA 12.8, bundles **triton 3.4.0**) | sm_80/86/89/90 supported. L4 = sm_89. |
| flash-attn | **prebuilt wheel**, no source build: `https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl` | The release lists both `cxx11abiTRUE` and `cxx11abiFALSE` wheels for torch 2.4–2.8. The M0 smoke test prints `torch._C._GLIBCXX_USE_CXX11_ABI`. If it is `False` or the import fails with `undefined symbol`, switch to the `cxx11abiFALSE` wheel. Second fallback: torch `2.9.1` + `flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl` from the v2.8.3 release. **[VERIFY at M0]** |
| transformers | `4.57.3` | Used for the HF reference, tokenizer and chat template. DFlash authors evaluated with 4.57.3. |
| others | `safetensors`, `xxhash`, `numpy`, `fastapi`, `uvicorn`, `httpx`, `pytest`, `huggingface_hub[hf_xet]` | |
| env | `HF_HUB_CACHE=/hf/hub`, `HF_XET_HIGH_PERFORMANCE=1`, `TOKENIZERS_PARALLELISM=false` | |

GPU constraints:
- FlashAttention-2 does **not** support Turing, so **never use T4**.
- In FA2, paged KV via `block_table` requires **page block size divisible by 256**. The FA backend therefore uses `block_size=256`, as nano-vllm does. The torch reference backend accepts any block size, and CPU tests use 16.

### 3.3 Modal baseline image (vLLM)
- `modal.Image.debian_slim(python_version="3.12").uv_pip_install("vllm==0.30.0")` (released 2026-09-22; pulls torch 2.13.0 + CUDA 13.0.3, flashinfer 0.6.18.post1).
  - CUDA 13.0 is within the host driver's CUDA 13.0, so it is OK. **[VERIFY at M8a smoke]**
- Do **not** use `vllm==0.31.0`. It was released 2026-10-05 and is too fresh.
- Fallback if 0.30.0 fails to start: use Modal's own tested example stack, `nvidia/cuda:12.9.0-devel-ubuntu22.04` + Python 3.12 + `vllm==0.21.0`. In that case DFlash support in vLLM may be missing; document it.
- vLLM DFlash flag: `--speculative-config '{"method":"dflash","model":"z-lab/Qwen3-4B-DFlash-b16","num_speculative_tokens":15}'`. If vLLM rejects it, report vLLM non-spec only and say so in the README.

### 3.4 Models (all ungated; **no HF token required**)
| Role | HF repo | Size |
|---|---|---|
| Dev target | `Qwen/Qwen3-0.6B` | 1.5 GB bf16, 28 layers, hidden 1024, 16 q heads / 8 kv heads, head_dim 128, tied embeddings |
| Spec target | `Qwen/Qwen3-4B` | 8 GB bf16, 36 layers, hidden 2560, 32 q / 8 kv heads, head_dim 128, tied embeddings, rope_theta 1e6 |
| Spec draft | `z-lab/Qwen3-4B-DFlash-b16` | 1.07 GB. 5 layers; `block_size=16`; `mask_token_id=151669`; `target_layer_ids=[1,9,17,25,33]`; shares target embed + lm_head |
| Optional | `Qwen/Qwen3-1.7B` | only for an extra non-spec benchmark point |

An HF token is only needed if anonymous downloads get rate-limited (HTTP 429). In that case the user creates a read token, runs `modal secret create hf-token HF_TOKEN=...`, and the download function attaches it. Otherwise do not use one.

---

## 4. Architecture

```
                 ┌──────────────────────────── one Modal L4 container (or local CPU) ─────────────────────────────┐
 HTTP client     │  server/api.py (FastAPI)                                                                       │
 (bench/client) ─┼─▶ /v1/completions  /v1/chat/completions  /v1/models  /health  /metrics                       │
   SSE stream  ◀─┼── per-request asyncio.Queue  ◀── call_soon_threadsafe ──┐                                      │
                 │                                                        │                                      │
                 │  server/async_engine.py: background thread: while True: engine.step()                         │
                 │                                                        │                                      │
                 │  engine.py  LLMEngine.step()                                                                   │
                 │     │ 1. scheduler.schedule() ──▶ Batch(kind=PREFILL | DECODE | SPEC, seqs)                   │
                 │     │ 2. model_runner.run(batch)    ──▶ token ids (+ aux hidden for SPEC)                     │
                 │     │ 3. scheduler.postprocess(seqs, new_tokens)  (append, stop checks, free blocks)          │
                 │     ▼                                                                                          │
                 │  scheduler.py (pure Python) ── uses ──▶ block_manager.py (pure Python: blocks, ref counts,    │
                 │                                          free list, prefix hash → block map)                  │
                 │                                                                                                │
                 │  model_runner.py  (torch; builds AttnMetadata: slot_mapping, block_tables, cu_seqlens;        │
                 │                    owns KV pools; CUDA graphs for DECODE)                                     │
                 │     ├── models/qwen3.py  Qwen3 target (+ capture aux hidden at layers [1,9,17,25,33])         │
                 │     ├── spec/dflash.py   DFlash draft (5 layers, non-causal block attention, own KV pool       │
                 │     │                    sharing the SAME block tables / slot mapping as the target)          │
                 │     └── attention/                                                                            │
                 │          ├── backend_flash.py  flash_attn_varlen_func / flash_attn_with_kvcache (GPU)        │
                 │          ├── backend_torch.py  reference paged attention in pure PyTorch (CPU tests, oracle)  │
                 │          └── triton_store.py   the ONE Triton kernel: scatter K/V into cache slots (optional) │
                 └────────────────────────────────────────────────────────────────────────────────────────────────┘

 KV cache layout (per layer): k_cache, v_cache : [num_blocks, block_size, num_kv_heads, head_dim]  (FA2 paged layout)
 slot = block_table[pos // block_size] * block_size + pos % block_size
```

### 4.1 File layout (create exactly this)
```
tinyserve/                      (repo root = C:\Users\tranb\projects\tinyserve)
├── PLAN.md  RESOURCES.md  README.md  LEARNING.md  RESULTS.md  COSTLOG.md  LICENSE (MIT)
├── pyproject.toml  .python-version  .gitignore  uv.lock
├── modal_app.py                # ALL Modal functions + local entrypoints (single file)
├── tinyserve/
│   ├── __init__.py             # exports LLM, SamplingParams
│   ├── config.py               # EngineConfig dataclass (+ validation)
│   ├── sampling.py             # SamplingParams; sample(logits, params) (greedy/temp/top-p/top-k)
│   ├── sequence.py             # Sequence, SequenceStatus
│   ├── block_manager.py        # Block, BlockManager, prefix hashing      (pure Python, NO torch import)
│   ├── scheduler.py            # Scheduler, Batch, BatchKind               (pure Python, NO torch import)
│   ├── engine.py               # LLMEngine + LLM (offline generate API)
│   ├── model_runner.py         # tensor prep, KV allocation, forward, CUDA graphs, spec step
│   ├── loader.py               # safetensors → module weight loading (name mapping, tied embeddings)
│   ├── attention/
│   │   ├── __init__.py         # get_backend(name) ; Attention nn.Module
│   │   ├── metadata.py         # AttnMetadata dataclass + set/get "current context"
│   │   ├── backend_torch.py
│   │   ├── backend_flash.py
│   │   └── triton_store.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── qwen3.py            # RMSNorm, RoPE, Qwen3Attention/MLP/DecoderLayer/Model/ForCausalLM
│   ├── spec/
│   │   ├── __init__.py
│   │   ├── proposer.py         # Proposer protocol; OracleProposer, RandomProposer (tests); DFlashProposer
│   │   ├── dflash.py           # DFlashDraftModel using tinyserve Attention
│   │   └── verify.py           # greedy_accept(draft_tokens, target_argmax) -> accepted counts (pure torch)
│   └── server/
│       ├── __init__.py
│       ├── async_engine.py     # thread + queues, request lifecycle, metrics
│       ├── protocol.py         # pydantic request/response models (OpenAI subset)
│       └── api.py              # FastAPI app factory; `python -m tinyserve.server.api --model ...`
├── bench/
│   ├── client.py               # async streaming load generator (works against tinyserve AND vLLM)
│   ├── workloads.py            # random-length, shared-prefix, spec (gsm8k/humaneval/mt-bench) workloads
│   ├── plot.py                 # results/*.json → results/charts/*.png
│   └── data/                   # small committed prompt files (jsonl) produced by scripts/prepare_data.py
├── scripts/
│   ├── prepare_data.py         # runs LOCALLY (free): writes bench/data/*.jsonl
│   ├── cost_guard.py           # reads `modal billing report`, refuses to continue past budget
│   └── record_result.py        # appends verified outputs to RESULTS.md with git SHA + timestamp
├── tests/
│   ├── conftest.py             # tiny random Qwen3 config factory (fp64), markers: slow, gpu
│   ├── reference/dflash_ref.py # vendored z-lab reference DFlash module (MIT, attributed) for parity tests
│   ├── cpu/                    # run locally, free
│   └── gpu/                    # run on Modal only, via modal_app.py::gpu_tests
└── results/
    ├── verify/                 # JSON written by verification runs
    ├── bench/                  # benchmark JSON
    └── charts/                 # PNGs referenced by README
```

### 4.2 Key data structures and interfaces (implement these signatures)

```python
# config.py
@dataclass
class EngineConfig:
    model: str = "Qwen/Qwen3-0.6B"
    dtype: str = "bfloat16"            # "float32"/"float64" allowed for CPU tests
    device: str = "cuda"               # "cpu" for local tests
    attn_backend: str = "flash"        # "flash" | "torch"
    block_size: int = 256              # flash backend asserts % 256 == 0; torch backend any >= 1
    num_kv_blocks: int | None = None   # None → derive from gpu_memory_utilization
    gpu_memory_utilization: float = 0.85
    max_num_seqs: int = 128
    max_num_batched_tokens: int = 8192
    max_model_len: int = 4096
    enable_prefix_cache: bool = True
    enforce_eager: bool = False        # True disables CUDA graphs
    cuda_graph_batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    spec_method: str | None = None     # None | "dflash" | "oracle" | "random" (last two: tests only)
    spec_draft_model: str | None = None  # "z-lab/Qwen3-4B-DFlash-b16"
    spec_block_size: int = 16          # DFlash block (1 real token + 15 mask tokens)
    use_triton_store: bool = True      # False → torch index_copy_ store (always False on CPU)
    seed: int = 0

# sampling.py
@dataclass
class SamplingParams:
    max_tokens: int = 256
    temperature: float = 0.0           # 0 → greedy
    top_p: float = 1.0
    top_k: int = -1
    ignore_eos: bool = False           # needed for fixed-length benchmarks (vLLM supports same flag)
    stop_token_ids: list[int] = field(default_factory=list)

# sequence.py
class SequenceStatus(Enum): WAITING, RUNNING, FINISHED
class Sequence:
    seq_id: int; status: SequenceStatus; params: SamplingParams
    token_ids: list[int]; num_prompt_tokens: int
    num_cached_tokens: int            # prompt tokens served from prefix cache (block-aligned)
    block_table: list[int]
    # timing (time.perf_counter): arrival, first_token, finish; per-token emit times for ITL
    # spec-only state:
    draft_ctx_len: int = 0            # positions [0, draft_ctx_len) have draft context KV written
    pending_hidden: "torch.Tensor | None" = None   # aux hidden for positions [draft_ctx_len, num_tokens-1)
    spec_steps: int = 0; spec_accepted: int = 0     # for τ = (accepted+steps)/steps
    @property num_tokens, num_completion_tokens, last_token, num_blocks(block_size)
    finish_reason: str | None         # "stop" | "length"

# block_manager.py   (NO torch import; must run on any machine)
class Block: block_id: int; ref_count: int; hash: int | None; token_ids: tuple[int, ...]
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_cache: bool): ...
    def can_allocate(self, seq: Sequence) -> bool
    def allocate(self, seq: Sequence) -> None
        # walks FULL prompt blocks; chained hash h_i = xxh64((h_{i-1}, tuple(tokens_i)));
        # on hit: reuse block (ref_count += 1) and add block_size to seq.num_cached_tokens;
        # first miss → stop matching; allocate fresh blocks for the rest.
        # INVARIANT: num_cached_tokens <= num_prompt_tokens - 1 (we must run >= 1 token to get logits).
        #   If the whole prompt is cached, drop the last hit block (recompute it).
    def deallocate(self, seq: Sequence) -> None      # ref_count -= 1; ref 0 → free list (hash KEPT for reuse,
                                                     #  evicted lazily when the block is reallocated = LRU-ish)
    def can_append(self, seq: Sequence, lookahead: int = 0) -> bool
    def ensure_slots(self, seq: Sequence, lookahead: int = 0) -> None
        # guarantees block_table covers positions [0, seq.num_tokens - 1 + lookahead]
        # (decode: lookahead=0 → slot for the last token; spec: lookahead = spec_block_size - 1)
    def commit_full_blocks(self, seq: Sequence, num_valid_kv_tokens: int) -> None
        # register hash for every block fully covered by VALID (committed) KV. In spec mode pass
        # min(target KV valid, seq.draft_ctx_len) so cached blocks always have valid draft KV too.
    @property num_free_blocks: int
    stats: dict  # {"prefix_query_tokens", "prefix_hit_tokens"} → hit rate

# scheduler.py   (NO torch import)
class BatchKind(Enum): PREFILL, DECODE, SPEC
@dataclass
class Batch: kind: BatchKind; seqs: list[Sequence]
class Scheduler:
    def __init__(self, cfg: EngineConfig, block_manager: BlockManager, eos_token_id: int): ...
    def add(self, seq: Sequence) -> None
    def schedule(self) -> Batch | None
        # 1) prefill-first: admit WAITING seqs while (num_seqs < max_num_seqs) and
        #    (sum uncached prompt tokens <= max_num_batched_tokens) and can_allocate.
        # 2) else DECODE (or SPEC if cfg.spec_method) over RUNNING: for each seq ensure_slots(lookahead);
        #    if out of blocks → preempt the most recently admitted running seq (free its blocks,
        #    reset token state to prompt+generated-so-far as a new prompt, push to FRONT of waiting).
    def postprocess(self, batch: Batch, new_tokens: list[list[int]]) -> list[Sequence]  # returns finished
        # appends 1 (prefill/decode) or 1..16 (spec) tokens; truncates at first EOS (unless ignore_eos)
        # or stop_token_id or max_tokens; frees finished seqs' blocks.
    def has_work(self) -> bool

# attention/metadata.py
@dataclass
class AttnMetadata:
    is_varlen: bool                 # True: PREFILL / SPEC-verify / DFlash-draft;  False: DECODE (1 query/seq)
    causal: bool = True             # DFlash draft uses causal=False
    cu_seqlens_q: Tensor | None = None   # int32 [B+1]
    cu_seqlens_k: Tensor | None = None   # int32 [B+1]  (FULL context length incl. cached prefix)
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: Tensor | None = None   # int64 [num_new_tokens]; -1 = do not store
    context_lens: Tensor | None = None   # int32 [B] (decode)
    block_tables: Tensor | None = None   # int32 [B, max_blocks]; None → K/V are the fresh contiguous tensors
def set_attn_metadata(m: AttnMetadata | None) -> None; def get_attn_metadata() -> AttnMetadata

# attention/__init__.py
class Attention(nn.Module):  # one per layer; holds references k_cache/v_cache assigned by model_runner
    def forward(self, q, k, v) -> Tensor   # q:[N,Hq,D] k,v:[N,Hkv,D]; 1) store k,v at slot_mapping; 2) attend
# backend_flash.py — exactly the nano-vllm call pattern:
#   varlen:  flash_attn_varlen_func(q, k_or_kcache, v_or_vcache, cu_seqlens_q=..., cu_seqlens_k=...,
#            max_seqlen_q=..., max_seqlen_k=..., softmax_scale=..., causal=m.causal, block_table=m.block_tables)
#   decode:  flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, cache_seqlens=m.context_lens,
#            block_table=m.block_tables, softmax_scale=..., causal=True)
# backend_torch.py — per sequence: gather K/V positions [0, k_len) via block_table, repeat_interleave for GQA,
#   explicit mask: query i (absolute pos k_len - q_len + i) may see key j iff (not causal) or j <= that pos;
#   torch.nn.functional.scaled_dot_product_attention or manual softmax. Slow but obviously correct.

# models/qwen3.py
class Qwen3ForCausalLM(nn.Module):
    def __init__(self, hf_config, aux_layer_ids: list[int] | None = None)
    def forward(self, input_ids: Tensor[N], positions: Tensor[N]) -> tuple[Tensor[N,H], Tensor[N, len(aux)*H] | None]
        # aux hidden = OUTPUT of decoder layer i (== HF output_hidden_states[i+1]) for i in aux_layer_ids,
        # concatenated on the last dim. NOTE: this is the residual stream (hidden + residual), pre-final-norm.
    def compute_logits(self, hidden: Tensor) -> Tensor   # final hidden is ALREADY normed; lm_head (tied → embed weight)
    embed_tokens, lm_head exposed (the DFlash draft reuses both)

# spec/proposer.py
class Proposer(Protocol):
    def propose(self, seqs: list[Sequence], runner) -> Tensor   # [B, spec_block_size-1] draft token ids
class OracleProposer   # tests: proposes the target's own greedy continuation (precomputed) → τ = 16
class RandomProposer   # tests: random tokens → τ ≈ 1
class DFlashProposer   # real

# spec/verify.py
def greedy_accept(draft: Tensor[B,K], target_argmax: Tensor[B,K+1]) -> tuple[Tensor[B] num_accepted, list[list[int]] new_tokens]
    # num_accepted[b] = length of the longest prefix where draft[b,j] == target_argmax[b,j];
    # new_tokens[b] = draft[b,:a] + [target_argmax[b,a]]   (always ≥ 1 token: the "bonus" token)
```

### 4.3 The DFlash spec step: exact semantics (the riskiest code; follow precisely)

Invariants for a RUNNING sequence with `n = seq.num_tokens` and last token `t = token_ids[n-1]` (position `n-1`):
- Target KV is valid for positions `[0, n-2]`. Position `n-1` (token `t`) has no KV yet. These are standard decode semantics.
- Draft context KV is valid for positions `[0, draft_ctx_len)`.
- `pending_hidden` holds target aux hidden for positions `[draft_ctx_len, n-2]`, shape `[n-1-draft_ctx_len, 5*H]`.

After **prefill** of a prompt of length L with c cached tokens:
- Target KV covers `[0, L-1]`. One token is sampled, so `n = L+1`.
- `pending_hidden` = aux hidden of the computed positions `[c, L-1]`.
- `draft_ctx_len = c`. Cached blocks already hold valid draft KV, guaranteed by the `commit_full_blocks` rule.

One **SPEC step** for a batch of B sequences, with K = 15 and block = 16:
1. **Slots.** `block_manager.ensure_slots(seq, lookahead=15)` gives slots for positions `[n-1, n+14]`.
2. **Draft forward** (one batched call, varlen):
   - Context part: `ctx = hidden_norm(fc(pending_hidden))`, shape `[*, H]`. For each draft layer ℓ:
     - `K_ctx = k_norm(k_proj_ℓ(ctx))` with RoPE at positions `[draft_ctx_len, n-2]`
     - `V_ctx = v_proj_ℓ(ctx)`
     - Store both into draft cache ℓ at those positions' slots.
     - **No queries come from context tokens.** Note that ctx is *not* passed through `input_layernorm`, and it is the same tensor for every layer.
   - Noise part: `x = target.embed_tokens([t, MASK×15])` at positions `[n-1, n+14]`. For each layer:
     - `h = input_layernorm(x)`; `q = q_norm(q_proj(h))`, `k = k_norm(k_proj(h))` (RoPE on both), `v = v_proj(h)`.
     - Store k, v at slots of `[n-1, n+14]`.
     - Attention with `causal=False`, `q_len = 16`, `k_len = n+15`, through `flash_attn_varlen_func(..., block_table=...)`.
     - `x = x + o_proj(attn)`; `x = x + mlp(post_attention_layernorm(x))`.
   - `draft_logits = target.lm_head(norm(x))[:, 1:16]` and `d_1..d_15 = argmax`.
   - Set `draft_ctx_len = n-1`. Noise KV at `[n-1, n+14]` is scratch and gets overwritten next step. This is the reference's `past_key_values_draft.crop(start)`.
3. **Verify** (target, varlen, causal):
   - Input `[t, d_1..d_15]` at positions `[n-1, n+14]`, `k_len = n+15`, with `block_table`.
   - Returns logits for all 16 positions plus aux hidden. `p = argmax(logits)`, shape `[16]`.
4. **Accept.** `a = greedy_accept(d, p)`. New tokens = `d_1..d_a, p_a` (`a+1` tokens, 1 ≤ a+1 ≤ 16).
5. **Commit.**
   - Append the tokens (scheduler.postprocess truncates at EOS/max_tokens).
   - `pending_hidden = aux[positions n-1 .. n-1+a]` (a+1 rows).
   - Target KV is now valid for `[0, n+a-1]` = `[0, n_new-2]`. Rejected KV slots beyond that are simply ignored and later overwritten.
   - `commit_full_blocks(seq, min(n_new-1, draft_ctx_len))`. Update `spec_steps += 1`, `spec_accepted += a`.
6. τ (mean acceptance length) = `(spec_accepted + spec_steps) / spec_steps`, which counts the bonus token. This matches the paper's τ definition, where τ includes the bonus token. **[VERIFY by reading the reference `acceptance_lengths.append(acceptance_length+1)`]**

**Correctness never depends on the draft.** A wrong draft only lowers τ. So the lossless test (spec output == non-spec output) can pass even with a buggy draft. The **τ checks** (parity with the reference implementation) are what catch draft bugs.

**Draft weight mapping** (z-lab safetensors keys → modules):
- `fc.weight [H, 5H]`, `hidden_norm.weight`, `norm.weight`
- `layers.{0..4}.{input_layernorm, post_attention_layernorm}.weight`
- `layers.i.self_attn.{q_proj,k_proj,v_proj,o_proj,q_norm,k_norm}.weight`
- `layers.i.mlp.{gate_proj,up_proj,down_proj}.weight`
- The draft has **no** embed or lm_head. It uses the target's.
- No biases. rope_theta 1e6, head_dim 128, 32 q heads / 8 kv heads.

---

## 5. Milestones (build in this order; do not start N+1 before N's check passes or is documented as failed)

Every milestone ends with:
- (a) running its **mechanical check**,
- (b) `uv run python scripts/record_result.py --milestone Mx --file <output>`, which appends to `RESULTS.md` with git SHA + UTC timestamp + raw output,
- (c) a focused git commit, and
- (d) for GPU milestones, a COSTLOG.md entry.

Legend: 🖥 = local CPU (free), ☁ = Modal L4 (costs money).

### M0: Scaffold + environment smoke (🖥 + ☁ ~0.25 GPU-h)
Build:
- `git init` inside the project folder.
- `.gitignore` (§10), `pyproject.toml`, `uv python pin 3.12`, `uv sync`.
- Empty package skeleton.
- `modal_app.py` containing:
  - `engine_image`, `vllm_image`, and volume `tinyserve-hf-cache`.
  - `download_models(repos: list[str])`: a **CPU-only** function (`cpu=2`, `memory=4096`, `timeout=1800`, no GPU) that `snapshot_download`s into the volume and calls `volume.commit()`.
  - `smoke()` on L4 (`timeout=600`). It prints torch/CUDA/FA versions, `torch._C._GLIBCXX_USE_CXX11_ABI`, GPU name, then runs one `flash_attn_with_kvcache` call with a 256-block `block_table` and one `flash_attn_varlen_func` call with `block_table`. It also runs a 4-line Triton kernel (vector add) to prove Triton JIT works.
- `scripts/cost_guard.py`: runs `modal billing report --for "this month" --json`, sums cost, exits non-zero if ≥ `$22` (warn) / `$25` (block). Every `local_entrypoint` that uses a GPU calls it first via `subprocess`. If the billing command fails, print a warning and ask the user to check the dashboard. Do not silently continue past repeated failures.

Check:
```powershell
uv run python -c "import sys, torch; print(sys.version); print(torch.__version__)"   # 3.12.x, 2.8.0+cpu
git rev-parse --show-toplevel                                                         # ...\projects\tinyserve
uv run modal run modal_app.py::download_models   # downloads Qwen3-0.6B, Qwen3-4B, z-lab/Qwen3-4B-DFlash-b16 (CPU only)
uv run modal run modal_app.py::smoke
```
Expected:
- `smoke` prints `FA_OK paged_kvcache max_abs_err<1e-2 vs torch reference`, `FA_VARLEN_BLOCKTABLE_OK`, `TRITON_OK`, and GPU `NVIDIA L4`.
- If FA import fails, follow the §3.2 fallback order and record which wheel worked.

### M1: Pure-Python core: block manager + scheduler + sampling (🖥 only, $0)
Build: `sequence.py`, `block_manager.py`, `scheduler.py`, `sampling.py`, `spec/verify.py`.

Tests in `tests/cpu/`:
- `test_block_manager.py`
  - allocate/deallocate ref counts
  - free-list conservation (`free + used == num_blocks` after every op)
  - `ensure_slots` with lookahead 0 and 15 across block boundaries
  - can_allocate false when full
  - prefix hits:
    - same prompt twice → second has `num_cached_tokens == floor((L-1)/bs)*bs`
    - divergent token in block k → hits only blocks < k
    - chained hash: identical block contents at different depths must NOT collide
    - whole-prompt-cached invariant
    - a freed block is reusable by hash until it is reallocated
  - 1,000-iteration randomized stress test (seeded): random allocate/append/free sequences, asserting invariants.
- `test_scheduler.py`
  - prefill admission respects `max_num_seqs` and `max_num_batched_tokens`
  - decode batches contain all running seqs
  - preemption when blocks run out (tiny pool) → the preempted seq goes to the front of waiting and later completes
  - EOS / max_tokens / ignore_eos / stop_token_ids truncation
  - spec postprocess appending 1..16 tokens with truncation in the middle of the accepted run
- `test_verify.py`
  - all-accept (a=15, 16 tokens)
  - none-accept (1 token)
  - first-mismatch-in-middle
  - batch with mixed results
- `test_sampling.py`: temperature 0 == argmax; top-k=1 == argmax; top-p mass property on a hand-made distribution; seeded determinism.

Check:
```powershell
uv run pytest tests/cpu/test_block_manager.py tests/cpu/test_scheduler.py tests/cpu/test_verify.py tests/cpu/test_sampling.py -q
uv run python -c "import ast,sys; [sys.exit(1) for f in ['tinyserve/block_manager.py','tinyserve/scheduler.py'] if 'torch' in open(f).read()]"
```
Expected: all pass (≥ 30 tests), and the second command exits with code 0 (no torch import in the pure-Python core).

### M2: Qwen3 model + attention backends + single-sequence correctness (🖥 then ☁ ~0.75 GPU-h)
Build:
- `models/qwen3.py`: RMSNorm, RoPE (neox-style rotate_half, theta from config, cos/sin cache), GQA attention with per-head `q_norm`/`k_norm` **before** RoPE, SwiGLU MLP, tied lm_head.
- `loader.py`: safetensors → params. Handle `tie_word_embeddings`. Optionally fuse qkv/gate_up; not required.
- `attention/` with the torch backend (any block size) and the flash backend (block_size % 256 == 0).
- Optional `triton_store.py`: one program per token, copies `num_kv_heads*head_dim` contiguous values from `k[i]`/`v[i]` to `k_cache.view(-1, D)[slot]`; skip if `slot == -1`. Write a docstring explaining program_id, `tl.arange`, masks and pointer arithmetic for a beginner.
- `model_runner.py`: PREFILL and DECODE paths for a single sequence first, then batched.
- `engine.py` `LLM.generate(prompts | token_ids, SamplingParams) -> list[dict(text, token_ids, ...)]`.

Tests:
- 🖥 `tests/cpu/test_model_tiny.py`: a tiny random Qwen3 (built from `Qwen3Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16, intermediate_size=128, vocab_size=512, tie_word_embeddings=True)`, `torch.manual_seed(0)`, **float64**). Save it to a temp dir with HF `save_pretrained`, then load the same weights into tinyserve. Check:
  - prefill logits max-abs-diff vs HF `Qwen3ForCausalLM` < 1e-9
  - greedy 32 tokens identical for 8 random prompts (torch backend, block_size 16)
- 🖥 `tests/cpu/test_qwen3_06b_cpu.py` (marker `slow`): real `Qwen/Qwen3-0.6B` in **float32** on CPU. 4 short prompts × 24 greedy tokens must be **identical** to HF `generate(do_sample=False)`. The download (~1.5 GB) is free.
- ☁ `tests/gpu/test_m2_gpu.py` (Qwen3-0.6B, bf16, L4):
  - (a) flash backend vs torch backend on identical inputs: prefill logits max-abs-diff < 0.15 and the top-1 matches on ≥ 99% of positions.
  - (b) Triton store == torch store (exact).
  - (c) Greedy 128 tokens vs HF (bf16, sdpa) on 8 prompts. **Near-tie rule** (used everywhere below): a prompt "matches" if the token IDs are identical, OR they diverge at a step where HF's top-1/top-2 logit gap is < 0.5. Require ≥ 7/8 exact and 8/8 match-or-near-tie.

Check:
```powershell
uv run pytest tests/cpu -q -m "not slow"; uv run pytest tests/cpu -q -m slow
uv run modal run modal_app.py::gpu_tests --suite m2
```
Expected: all CPU tests pass. The GPU suite prints `M2 PASS` with the numbers and writes `results/verify/m2.json` locally (the function returns JSON and the entrypoint writes it).

### M3: Paged KV + continuous batching + preemption (🖥 then ☁ ~1.0 GPU-h)
Build:
- Full engine loop.
- Batched PREFILL varlen with `cu_seqlens`.
- Batched DECODE with `block_tables`/`context_lens`.
- KV pool sizing from `torch.cuda.mem_get_info()` after a warmup forward at `max_num_batched_tokens`, as nano-vllm does.
- Preemption by recompute.

Tests:
- 🖥 tiny fp64 model, torch backend, `block_size=16`:
  - (a) 24 prompts of random lengths 5–200 with `max_num_seqs=8` give outputs identical to running each prompt alone.
  - (b) A pool of only 12 blocks forces preemption: all requests finish, outputs are identical to (a), and the test asserts preemption happened (counter > 0).
  - (c) Random arrival: add requests between `step()` calls; outputs are unchanged.
- ☁ Qwen3-0.6B bf16:
  - (a) 64 prompts (`bench/data/mixed_prompts.jsonl`), batched with `max_num_seqs=32` vs bs=1 sequential: ≥ 60/64 exact, 64/64 exact-or-near-tie.
  - (b) Offline throughput sanity: 256 prompts, in 512 / out 256 with `ignore_eos`. Print tok/s. No threshold yet; record it.

Check:
```powershell
uv run pytest tests/cpu/test_engine_batching.py -q
uv run modal run modal_app.py::gpu_tests --suite m3
```

### M4: Prefix caching (🖥 then ☁ ~0.5 GPU-h)
Build:
- Wire `BlockManager` prefix hits into prefill. Uncached tokens only go through the model, using `flash_attn_varlen_func` with `block_table` and `cu_seqlens_k` = full length.
- Add `/metrics`-ready counters.

Tests:
- 🖥 tiny model, block 16: 10 requests sharing a 100-token prefix:
  - outputs identical with cache on vs off
  - hit-rate = hit_tokens / prompt_tokens ≥ 0.75 (exact expected value computed in the test from block math)
  - refcount returns to 0 after all finish
- ☁ Qwen3-0.6B, block 256, workload W2 (32 requests, 2048-token shared prefix + 64-token unique suffix, 64 output tokens; first request sequential, then the rest at concurrency 8):
  - (a) outputs identical cache on vs off (≥ 31/32 exact, all exact-or-near-tie)
  - (b) hit rate ≥ **0.85**
  - (c) TTFT p50 with cache ≤ **0.5×** TTFT p50 without cache

Check: `uv run modal run modal_app.py::gpu_tests --suite m4` → `M4 PASS hit_rate=… ttft_speedup=…`.

### M5: CUDA graphs for decode (☁ ~0.75 GPU-h)
Build:
- Capture one graph per batch size in `cuda_graph_batch_sizes` (≤ `max_num_seqs`) with static input buffers: input_ids, positions, slot_mapping, context_lens, block_tables (max blocks), and output logits/hidden.
- At replay, pad the batch to the next captured size. Padded rows use `slot_mapping=-1` and `context_lens=1`, which nano-vllm-style kernels tolerate. **[VERIFY: FA with context_len for padded rows; if it errors, point padded rows at a reserved dummy block 0 that is never allocated]**
- Share the memory pool across graphs (`torch.cuda.graph(pool=...)`).
- `enforce_eager=True` bypasses graphs.

Tests (☁ Qwen3-0.6B):
- (a) 32 prompts × 128 tokens, graph vs eager: ≥ 31/32 exact, all exact-or-near-tie.
- (b) bs=1 decode ITL p50: graph ≤ **0.67×** eager (≥ 1.5× faster).
- (c) bs=32 decode step time: graph ≤ eager.

Check: `uv run modal run modal_app.py::gpu_tests --suite m5`.

### M6: OpenAI-compatible streaming server (🖥 then ☁ ~0.5 GPU-h)
Build:
- `server/protocol.py`, `async_engine.py`, `api.py`.
- `POST /v1/completions`:
  - `prompt` is `str | list[int]`
  - accepts `max_tokens`, `temperature`, `top_p`, `stream`, `ignore_eos`, `seed`
- `POST /v1/chat/completions`:
  - `messages`, plus `chat_template_kwargs` (default `{"enable_thinking": False}`), applied via the HF tokenizer chat template
- `GET /v1/models`, `/health`, `/metrics` (JSON: running, waiting, kv_usage, prefix_hit_rate, spec_tau).
- Stream format: `data: {json}\n\n` chunks, then `data: [DONE]\n\n`. The last chunk includes `usage` when `stream_options.include_usage` is set.
- Incremental detokenization: decode the full output each time and emit the new suffix. Hold back trailing U+FFFD (partial UTF-8).
- The engine thread only steps when there is work: block on a `threading.Event`, never busy-spin.
- Launch: `python -m tinyserve.server.api --model Qwen/Qwen3-0.6B --port 8000`.

Tests:
- 🖥 `tests/cpu/test_server.py`: FastAPI `TestClient` with the tiny random model and a tiny tokenizer. Use the real Qwen3 tokenizer files, downloaded locally for free; they are small.
  - non-stream and stream responses parse
  - `[DONE]` terminator
  - 8 concurrent requests via `httpx.AsyncClient` complete
  - streamed text == non-streamed text for greedy
- ☁ Server and `bench/client.py` run **in the same container**: start uvicorn as a subprocess on 127.0.0.1, poll `/health`, run the client, then kill the server. No Modal web endpoint and no deployment.
  - The official `openai` Python client streaming works (add `openai` to the image).
  - Concurrency 32 × 64 requests: 0 errors, and TTFT/ITL printed.

Check: `uv run pytest tests/cpu/test_server.py -q` then `uv run modal run modal_app.py::gpu_tests --suite m6`.

### M7: DFlash speculative decoding (headline) (🖥 then ☁ ~2.5 GPU-h)
Build:
- `spec/dflash.py`. The draft attention reuses the `Attention` module with `causal=False` and its own KV pool of 5 layers on the **same** block tables.
- `spec/proposer.py`.
- The SPEC path in `model_runner.py` and `scheduler.py`, exactly as in §4.3.
- Aux-hidden capture in the target. The prefill path also returns aux hidden, sliced per sequence.
- `EngineConfig.spec_method`.
- Vendor the reference module into `tests/reference/dflash_ref.py`:
  - copy from `https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16/raw/main/dflash.py` + `utils.py`
  - MIT; keep the copyright/attribution header
  - make imports local
  - it depends on `transformers==4.57.3` internals

Tests:
- 🖥 tiny fp64 target (4 layers) + tiny random DFlash draft (2 layers, `target_layer_ids=[0,2]`, block 16, `mask_token_id` = vocab_size-1), torch backend, block_size 16:
  1. **Draft parity:** with weights copied into the reference module, draft logits for a given context and block have max-abs-diff < 1e-9.
  2. **Lossless:**
     - spec output == non-spec output for 16 prompts × 64 tokens at batch 8, for each of `oracle`, `random` and `dflash` (random-init) proposers.
     - oracle gives τ == 16.0 except at sequence end.
     - random gives τ ≤ 1.1.
  3. **Reference loop parity:** tinyserve spec (bs=1) produces the **same acceptance-length sequence** per step as `dflash_ref.spec_generate` on 4 prompts, which proves the KV/position bookkeeping matches.
  4. Spec with preemption (tiny pool) is still lossless.
  5. Prefix cache + spec is still lossless, and blocks are committed only when draft KV is valid.
- ☁ `tests/gpu/test_m7_gpu.py`: Qwen3-4B + `z-lab/Qwen3-4B-DFlash-b16`, bf16, L4. Use `bench/data/{gsm8k,humaneval,mtbench}.jsonl` (32 prompts each), chat template with `enable_thinking=False`, greedy, `max_tokens=512`. Run in one container:
  - (a) **τ parity with reference:** run `dflash_ref.spec_generate` (HF target, sdpa) on the first 8 prompts of each dataset. tinyserve τ (bs=1) must be within **±10%** of the reference τ per dataset.
  - (b) **τ vs paper:** tinyserve τ ≥ **0.7×** the paper's value (GSM8K ≥ 4.5, HumanEval ≥ 4.6, MT-Bench ≥ 3.0). Print the ratio. We use `max_tokens=512` vs the paper's 2048, so a mild mismatch is expected and must be explained.
  - (c) **Lossless:** spec vs non-spec tinyserve outputs: ≥ 90/96 exact, 96/96 exact-or-near-tie.
  - (d) **Speedup:** decode tokens/s at bs=1 on gsm8k, spec ÷ non-spec (non-spec **with** CUDA graphs) ≥ **2.0×**. Also report the c = 4 and c = 16 numbers. The paper reports 5.15× on H200 with a tuned stack, and our eager spec path on an L4 will be slower; record it honestly.

Check:
```powershell
uv run pytest tests/cpu/test_dflash_*.py -q
uv run modal run modal_app.py::gpu_tests --suite m7
```
Debug checklist if τ is low (≈1–2) while lossless passes:
- the target layer offset (aux = output of layer i, i.e. HF `hidden_states[i+1]`)
- fc → hidden_norm order
- k_norm applied to ctx keys
- RoPE positions for ctx vs noise
- MASK id 151669
- the draft uses the **target's** embed and lm_head
- `causal=False`
- thinking disabled in the prompt
- draft logits taken from block positions 1..15, not 0..14

### M8: Benchmarks vs vLLM + charts (☁ ~2.75 GPU-h)
Build:
- `bench/client.py`: async httpx, closed-loop concurrency, streaming. It records per request: send time, first-token time, every chunk time, output token count (from `usage`), and errors.
- `bench/workloads.py` and `bench/plot.py`.
- Modal functions:
  - `bench_tinyserve(cfg)`: server subprocess + client in one container
  - `bench_vllm(cfg)`: `vllm serve` subprocess + the same client in one container; `vllm_image`; `timeout=2400`
  - Both return JSON. The local entrypoint writes `results/bench/*.json`.

Protocol (§7) and check:
```powershell
uv run modal run modal_app.py::bench --engine tinyserve --suite core
uv run modal run modal_app.py::bench --engine vllm --suite core
uv run modal run modal_app.py::bench --engine tinyserve --suite spec
uv run modal run modal_app.py::bench --engine vllm --suite spec
uv run python bench/plot.py      # writes results/charts/*.png
uv run python scripts/record_result.py --milestone M8 --file results/bench/summary.md
```
Targets (report regardless of outcome):
- tinyserve offline throughput at c = 64 on Qwen3-0.6B ≥ **60%** of vLLM (stretch ≥ 85%).
- tinyserve-DFlash bs=1 speedup ≥ 2.0× over tinyserve non-spec.
- tinyserve-DFlash τ within ±10% of vLLM-DFlash τ, if vLLM DFlash runs.

Missing a target is acceptable. **Claiming** to hit it without the artifact is not.

### M9: Stretch (only if all of M0–M8 are done AND spend < $18)
Pick at most one:
- (a) chunked prefill (Sarathi-Serve) with an ITL-under-load chart
- (b) SuffixDecoding-style model-free proposer via the same `Proposer` interface
- (c) CUDA graphs for the spec verify step
- (d) a 1-hour H100 rerun of the key charts (≈ $4.3)

---

## 6. Testing strategy (maximize free CPU tests; minimize GPU minutes)

1. **Pure-Python core.** The block manager and scheduler import no torch, so they can be unit tested in milliseconds on any machine. This is where most engine bugs live (off-by-one slots, ref counts, preemption).
2. **A torch reference attention backend.** It runs the *entire engine* on CPU with any block size. Tiny random Qwen3 models in **float64** make greedy decoding deterministic and free of ties, so CPU tests can demand *exact* equality:
   - batched == sequential
   - cache on == cache off
   - spec == non-spec
   - preempted == not preempted
3. **Real-model CPU parity.** Qwen3-0.6B in fp32 on the laptop checks the model math against HF for free (marker `slow`).
4. **GPU tests answer only the GPU-specific questions:**
   - FA kernel parity vs the torch backend
   - the Triton store
   - CUDA graphs
   - bf16 behaviour (near-tie rule)
   - real-draft τ
   - speed
   They run **as one pytest invocation per milestone inside one container** (`gpu_tests --suite mX`), never one Modal call per test, to avoid repeated cold starts and model loads.
5. **The near-tie rule** is the only allowed relaxation, and it is defined once in `tests/util.py`. Any divergence that is *not* a near-tie is a bug.
6. **Mock proposers** (oracle/random) test the spec plumbing independently of draft quality.
7. **Mark GPU tests** with `@pytest.mark.gpu` and skip them automatically when `not torch.cuda.is_available()`.

---

## 7. Benchmark protocol

- **Hardware:** Modal `gpu="L4"` (single type, no fallback list). Record `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv` in every result JSON.
- **Fairness:**
  - same model and dtype (bf16), `max_model_len=4096`, GPU memory utilization 0.85
  - prefix caching ON for both by default
  - vLLM with default CUDA graphs (no `--enforce-eager`), `--max-num-seqs 128` to match tinyserve
  - The server and client run in the same container for both engines, with the same client code and the same request payloads.
  - Use `/v1/completions` with token-ID prompts (exact lengths) and `ignore_eos: true` for the throughput workloads. Use `/v1/chat/completions` + `chat_template_kwargs: {"enable_thinking": false}` + `temperature: 0` for spec workloads.
- **Workloads** (generated deterministically with seed 0 and committed to `bench/data/`):
  - **W1 random:** 256 requests, input length uniform [256, 768], output fixed 256 (`ignore_eos`). Concurrency ∈ {1, 4, 16, 64, 128}. Model Qwen3-0.6B. Optionally one point (c = 64) on Qwen3-4B.
  - **W2 shared-prefix:** 8 groups × 16 requests. Each group shares a 2048-token prefix plus a 64–128-token unique suffix; output 128. Concurrency 16. Run cache ON vs OFF for tinyserve, and ON for vLLM.
  - **W3 spec:** gsm8k / humaneval / mt-bench, 32 prompts each, greedy, `max_tokens=512`, Qwen3-4B. Concurrency ∈ {1, 2, 4, 8, 16}. Four configs: tinyserve non-spec, tinyserve DFlash, vLLM non-spec, vLLM DFlash.
  - (Optional W4) 200 ShareGPT conversations (`anon8231489123/ShareGPT_Vicuna_unfiltered`, ungated): first human turn, natural output lengths capped at 512.
- **Warmup:** 8 requests, discarded. **Repeats:** 1 per point, plus a 2nd run at c = 16 (W1) and c = 1 (W3) to report run-to-run spread.
- **Metrics per point:**
  - TTFT p50/p90/p99
  - ITL p50/p99 (gaps between streamed chunks; for spec decoding, chunks may carry several tokens, so also report)
  - TPOT = (e2e − TTFT)/(output_tokens − 1), p50/p99
  - output tokens/s, requests/s, error count
  - for spec: τ (from `/metrics` for tinyserve; from vLLM's spec-decode metrics/logs for vLLM)
- **JSON schema** (`results/bench/{engine}_{model}_{workload}_{gpu}_{yyyymmdd}.json`):
```json
{"engine":"tinyserve","engine_version":"<git sha>","model":"Qwen/Qwen3-0.6B","gpu":"NVIDIA L4","driver":"580.95.05",
 "workload":"W1","config":{...},"points":[{"concurrency":16,"ttft_ms":{"p50":..,"p90":..,"p99":..},
 "itl_ms":{...},"tpot_ms":{...},"output_tok_s":..,"req_s":..,"errors":0,"tau":null}],"started_utc":"...","wall_s":...}
```
- **Charts** (`bench/plot.py`, matplotlib, one PNG each, labelled axes and units, engine legend):
  1. `throughput_vs_concurrency.png` (W1)
  2. `ttft_p50_vs_concurrency.png` and `tpot_p50_vs_concurrency.png` (W1)
  3. `prefix_cache_ttft.png` (W2 bars: off/on/vLLM)
  4. `spec_speedup_vs_concurrency.png` (W3: tinyserve-DFlash ÷ tinyserve-base, vLLM-DFlash ÷ vLLM-base)
  5. `tau_vs_paper.png` (W3: per-dataset τ for tinyserve, reference HF, vLLM, paper)
  6. `cuda_graph_itl.png` (M5)
- `results/bench/summary.md`: auto-generated table of key numbers, which the README embeds.

---

## 8. Modal budget and cost control

### 8.1 Prices (modal.com/pricing, checked 2026-10-06; per-second billing)
| Resource | $/sec | $/hour | Notes |
|---|---|---|---|
| T4 | 0.000164 | 0.59 | **Unusable**: no FA2 on Turing |
| **L4 (24 GB)** | 0.000222 | **0.80** | **Default for everything** |
| A10 (24 GB) | 0.000306 | 1.10 | Fallback only if L4 capacity is unavailable |
| L40S (48 GB) | 0.000542 | 1.95 | not needed |
| A100 40/80 GB | 0.000583 / 0.000694 | 2.10 / 2.50 | not needed |
| H100 | 0.001097 | 3.95 | optional M9(d), 1 hr max |
| CPU | 0.0000131 /core | 0.047 /core | request `cpu=4` on GPU fns → $0.19/h |
| Memory | 0.00000222 /GiB | 0.008 /GiB | request `memory=16384` (16 GiB) → $0.13/h; Qwen3-4B load needs ~10 GiB host RAM |
| Volumes | — | — | $0.09/GiB-month; **1 TiB free**. Our weights ≈ 12 GB, so $0 |
| Free credits | — | — | Starter plan: **$30/month** |

**Effective L4 container cost ≈ $1.12/hour** (GPU + 4 CPU + 16 GiB).

### 8.2 GPU-hour budget
| Item | L4-hours | ≈ $ |
|---|---|---|
| M0 smoke (+ image build checks) | 0.25 | 0.28 |
| M2 model correctness | 0.75 | 0.84 |
| M3 batching/preemption | 1.0 | 1.12 |
| M4 prefix cache | 0.5 | 0.56 |
| M5 CUDA graphs | 0.75 | 0.84 |
| M6 server | 0.5 | 0.56 |
| M7 DFlash (incl. HF reference τ runs) | 2.5 | 2.80 |
| M8 benchmarks (tinyserve core 0.5, vLLM core 0.5, spec ×2 engines 1.5, prefix 0.25) | 2.75 | 3.08 |
| Debug contingency | 3.0 | 3.36 |
| CPU-only functions (downloads, data) | — | ~0.30 |
| **Planned total** | **~12 L4-h** | **≈ $13.8** |
| Optional M9(d) H100 1 h (+CPU/mem) | 1.0 H100-h | ≈ 4.3 |
| **Grand total incl. optional** | | **≈ $18** |

**Thresholds:**
- **Warn at $22.**
- **Hard stop at $25.** Stop GPU work, finish the docs with existing results, and record the overrun reason in COSTLOG.md.
- Leave ≥ $5 headroom under the $30 cap for billing lag.

### 8.3 Cost-control rules (mandatory; the implementer must follow every one)
1. **Never `modal deploy`. Never use `modal serve` for long. Never set `min_containers`.** Use only `modal run`, whose ephemeral app stops when the local entrypoint returns.
2. Every `@app.function` has an explicit `timeout`:
   - downloads: 1800
   - smoke: 600
   - gpu_tests: 1800
   - bench: 2400
3. Also on every function: `scaledown_window=30` (or the minimum Modal allows), `max_containers=1`, and a **single GPU type**: `gpu="L4"` with no fallback list.
4. **Weights live in the Modal Volume `tinyserve-hf-cache`**, mounted at `/hf`. GPU functions must never download. They set `HF_HUB_OFFLINE=1` and fail fast if a model is missing.
5. **Images are built once.** Code is attached with `.add_local_python_source("tinyserve")` and `.add_local_dir(...)` as the **last** image steps (runtime mounts), so code edits do not rebuild images. Do not change pip pins casually; each change triggers a rebuild.
6. **One container per milestone check:** all GPU tests for a milestone run in a single pytest call, and model loads are shared through session-scoped fixtures.
7. **Do not iterate on GPU.** Reproduce failures on CPU with the tiny model first. Only go back to the GPU once the CPU repro passes.
8. The `cost_guard.py` pre-check runs before every GPU entrypoint.
9. After every session: `uv run modal app list` must show no running `tinyserve` apps. Stop stragglers with `modal app stop <id>`.
10. Do **not** use `modal run --detach`. If the laptop disconnects, the run should die.
11. Write a **COSTLOG.md** row for every GPU run. Template:

```markdown
| date (UTC) | command | GPU | wall min | est $ (wall×$1.12/60) | billing month-to-date $ (`modal billing report --for "this month" --json`) | purpose / outcome |
```

Also paste `uv run modal billing report --for today --show-resources` output at the end of each working day.

---

## 9. Definition of Done (every item needs a command or artifact) + anti-premature-done protocol

### 9.1 Checklist
- [ ] `uv run pytest tests/cpu -q -m "not slow"` passes. Output pasted in RESULTS.md (≥ 50 tests).
- [ ] `uv run pytest tests/cpu -q -m slow` passes (Qwen3-0.6B fp32 HF parity).
- [ ] `results/verify/m2.json` … `m7.json` exist, and each has `"status": "PASS"` or an explicit `"status": "FAIL"` with a README "What didn't work" entry.
- [ ] `results/bench/` contains W1, W2 and W3 JSON for tinyserve **and** vLLM (or a documented vLLM failure).
- [ ] `results/charts/` contains the 6 PNGs from §7, and README embeds them.
- [ ] README states the measured τ per dataset next to the paper and reference values.
- [ ] README states measured throughput as a % of vLLM, with GPU, versions and date.
- [ ] `python -m tinyserve.server.api` works with the `openai` client streaming example shown in the README (the transcript is pasted in RESULTS.md from the M6 GPU run).
- [ ] COSTLOG.md is complete, and the total month-to-date from `modal billing report` is ≤ $25.
- [ ] `uv run modal app list` shows no running tinyserve apps (paste the output).
- [ ] `git log --oneline` shows ≥ 15 focused commits; `git status` is clean; `git ls-files | Select-String -Pattern "\.env|safetensors|\.bin$|token"` returns nothing.
- [ ] LEARNING.md is complete per §12. README is complete per §11, including "What didn't work / didn't reproduce" (non-empty; at minimum it discusses bf16 near-ties and τ vs paper).
- [ ] LICENSE (MIT). The README credits nano-vllm, vLLM, SGLang/mini-sglang, flash-attn and z-lab/DFlash.

### 9.2 Anti-premature-done protocol
1. **No claim without an artifact.** Every number in README/RESULTS must come from a file in `results/` or a RESULTS.md entry written by `scripts/record_result.py`. That script stamps the git SHA, UTC time, exact command and raw stdout. It must refuse to run on a dirty tree for GPU results, unless `--allow-dirty` is passed, which is noted in the entry.
2. **Verify scripts print `PASS`/`FAIL` with the measured value and threshold**, e.g. `M4 hit_rate=0.91 (>=0.85) PASS`. Never print just "done".
3. **Do not weaken thresholds silently.** If a threshold is missed:
   - keep it
   - mark FAIL
   - write the analysis in README "What didn't work / didn't reproduce" (hypothesis, evidence, what you'd try next)
   - continue to the next milestone if the remaining work does not depend on it
4. **Do not mark a test skipped or xfail** to get a green run unless it is gpu-only on CPU. List any xfail in RESULTS.md with the reason.
5. **Before declaring the project done**, re-run `uv run pytest tests/cpu -q` from a clean clone (`git clone . ../tinyserve-verify; cd ../tinyserve-verify; uv sync; uv run pytest tests/cpu -q -m "not slow"`) and paste the output.
6. **Final message to the user** must list every DoD item with ✅ / ❌ and the artifact path. ❌ items need a one-line reason.

---

## 10. Git hygiene
- `git init` in the project folder (M0). The home directory is a separate repo.
- `.gitignore`:
```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.env
.env.*
*.safetensors
*.bin
*.pt
*.pth
hf_cache/
.cache/
models/
.modal/
*.log
.DS_Store
Thumbs.db
results/tmp/
```
- Never commit secrets. The Modal token lives in `~/.modal.toml`, outside the repo. An HF token, if ever needed, lives only in a Modal Secret. Never print tokens in logs.
- Commit `uv.lock`. Commit small data files in `bench/data/` (< 2 MB total, with attribution in `bench/data/README.md`: GSM8K MIT, HumanEval MIT, MT-Bench prompts per HF card).
- Small focused commits, conventional style, e.g.:
  - `feat(block_manager): chained-hash prefix caching`
  - `test(scheduler): preemption under block pressure`
  - `bench: W1 vLLM baseline on L4`
  - `docs: README results section`
  - one or more commits per milestone; never one giant commit
- Use the owner's git identity as already configured. Do not change git config.

---

## 11. README outline (blog-style; written last, from artifacts only)
1. **Title + one-line pitch:** "tinyserve: a ~3k-line LLM inference engine with paged KV, continuous batching, prefix caching, CUDA graphs and 2026 DFlash speculative decoding, benchmarked against vLLM on a $0.80/hr GPU."
2. **Headline results box:** 3–4 numbers (throughput % of vLLM, DFlash speedup at bs = 1, τ vs paper, prefix-cache TTFT win) plus the GPU and date.
3. **Quickstart:** `uv sync`, `uv run pytest tests/cpu`, the Modal commands, and an `openai` client snippet.
4. **Architecture:** the ASCII diagram from §4 and the life of a request.
5. **Why each optimization helps** (one section each: the problem, the idea, how tinyserve implements it, the measured effect + chart):
   - paged KV (fragmentation, block tables)
   - continuous batching (head-of-line blocking, Orca)
   - prefix caching (hash chain vs radix tree)
   - CUDA graphs (CPU launch overhead in decode)
   - DFlash (memory-bound decode → verify many tokens per weight read; why parallel block drafting beats autoregressive drafting; why the speedup shrinks with concurrency)
6. **Benchmarks vs vLLM:** methodology (§7), charts, and the tables from `summary.md`.
7. **Correctness:** how losslessness is tested (fp64 exactness, the near-tie rule in bf16, oracle/random proposers).
8. **What didn't work / didn't reproduce:** mandatory and honest.
9. **Cost:** the total Modal spend from COSTLOG and the cost-control tricks used.
10. **Differences from nano-vllm**, stated explicitly:
    - (a) DFlash spec decoding integrated with paged KV + batching
    - (b) an OpenAI streaming server and latency (TTFT/ITL) metrics
    - (c) a head-to-head vs vLLM on the same GPU, including vLLM's own DFlash
    - (d) a CPU-runnable reference backend with an exact fp64 test suite
    - (e) a cost-disciplined serverless GPU harness
    - (f) the write-up
    - Credit nano-vllm for the design inspiration (block manager + flash-attn call pattern).
11. **Future work:** the non-goals list, plus disaggregated P/D, chunked prefill, and EAGLE-3 comparison.
12. **References:** link to RESOURCES.md.

## 12. LEARNING.md outline (plain English, for interviews)
For each concept: **what problem it solves → intuition/analogy → the math or numbers → where it lives in tinyserve (file:function) → 3–5 likely interview questions with short model answers.**
1. Transformer inference phases: prefill (compute-bound) vs decode (memory-bandwidth-bound). Arithmetic intensity, and a roofline sketch with L4 numbers (~300 GB/s, ~121 dense bf16 TFLOPs; **[VERIFY spec sheet]**).
2. KV cache: what it stores, the size formula `2 × layers × kv_heads × head_dim × bytes × tokens` (worked example for Qwen3-0.6B and 4B), GQA.
3. PagedAttention: fragmentation, block tables, slot mapping, copy-on-write / ref counts.
4. Continuous batching (Orca) vs static batching; preemption (recompute vs swap).
5. Prefix caching: hash chain (vLLM APC) vs radix tree (SGLang RadixAttention); eviction; why block size 256 hurts hit granularity.
6. FlashAttention: tiling, online softmax, why it's IO-aware; varlen + paged variants.
7. CUDA graphs: kernel launch overhead, static shapes, padding to bucket sizes.
8. Speculative decoding: the draft/verify loop; why greedy verification is lossless; expected speedup vs τ and draft cost; rejection sampling for temperature > 0 (concept only).
9. EAGLE → EAGLE-3 → DFlash: feature-level drafting, training-time test, block diffusion drafting, KV injection of target features.
10. Serving metrics: TTFT, ITL, TPOT, throughput, goodput, SLOs; the latency/throughput tradeoff vs concurrency.
11. Disaggregated prefill/decode (DistServe, Mooncake) and chunked prefill (Sarathi): concepts only.
12. How Modal / Nebius / Baseten serve models: cold starts, weight caching in volumes, autoscaling, cost per token. Include a worked cost-per-1M-tokens calculation from our benchmark numbers.
13. **Interview question bank** (≥ 30 questions), e.g.:
    - "Why is decode memory-bound?"
    - "What happens when the KV cache is full?"
    - "How would you pick the speculative block size?"
    - "Why does spec decoding help less at high batch size?"
    - "How do you test an inference engine for correctness when bf16 is nondeterministic?"
    - "Walk me through a request's life in your engine."
    - "How does prefix caching interact with spec decoding in your design?"

---

## 13. Risks (where a one-shot build is most likely to fail) and mitigations

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| R1 | flash-attn wheel ABI/CUDA mismatch → import error | Med | M0 smoke catches it in minutes; documented 2-step fallback (§3.2); the torch backend still runs on GPU (slow) as a last resort so other milestones are not blocked. |
| R2 | FA paged KV needs block_size % 256; varlen+block_table edge cases | Med | Use 256 like nano-vllm; GPU parity test vs the torch backend (M2a). Prefix caching at 256 granularity is a known, documented limitation. |
| R3 | bf16 nondeterminism makes "token-identical" fail on GPU | High | Exactness proven on CPU in fp64; GPU uses the single, pre-declared near-tie rule; divergences are logged with logit gaps. |
| R4 | DFlash bookkeeping bugs (positions, draft KV, aux layer offset) | **High** | §4.3 spells out every invariant; a vendored reference + per-step acceptance-sequence parity test on CPU; τ parity vs HF reference on GPU; debug checklist in M7. |
| R5 | Spec + preemption + prefix cache interactions | Med | `commit_full_blocks(min(target_valid, draft_ctx_len))` rule; dedicated CPU tests M7.4/M7.5. Correctness never depends on draft KV (only τ does). |
| R6 | CUDA graph stale buffers / padded rows | Med | Graph vs eager equality test; `enforce_eager` escape hatch; dummy block for padded rows. |
| R7 | Budget overrun from hangs, retries or warm containers | Med | Timeouts everywhere, max_containers=1, no deploy, cost_guard, COSTLOG, CPU-first debugging. |
| R8 | vLLM 0.30 install, startup or DFlash flag fails on Modal | Med | M8 smoke with 1 request first (≤ 5 min); fallback to Modal's tested stack (cuda 12.9 + vLLM 0.21.0); document it. |
| R9 | Server thread/async deadlocks; SSE format mistakes | Med | CPU TestClient tests incl. concurrency; the official `openai` client test on GPU. |
| R10 | Implementer over-scopes (chunked prefill, tree drafts, torch.compile) | Med | Non-goals list; M9 gated on budget and completion. |
| R11 | Qwen3 chat template / thinking mode mismatch lowers τ | Med | Always `enable_thinking=False` for W3; prompts produced once by `prepare_data.py` and shared by all engines. |
| R12 | Anonymous HF download rate limits | Low | One-time CPU download into the volume; optional `hf-token` Modal secret. |
| R13 | Windows-specific issues (paths, encodings, no triton/flash-attn) | Med | pathlib everywhere; `encoding="utf-8"` on all file writes; lazy GPU-only imports. |
| R14 | Performance far below vLLM | High-ish | Expected for a ~3k-line engine without fused kernels or overlap scheduling; report honestly; profile with `torch.profiler` once (≤ 10 min GPU) and list the top 3 bottlenecks in README. |

---

## 14. What the user must provide / decide
- Nothing is required to start. A Modal token is already configured; all models and datasets are ungated, so no HF token is needed.
- Optional: an HF read token, only if downloads hit HTTP 429 (`modal secret create hf-token HF_TOKEN=...`).
- Approval to spend up to ~$14 (core) / ~$18 (with an optional H100 hour) of the $30 monthly Modal credits.
- Optional: whether to make the GitHub repo public at the end. Creating the remote and pushing is the user's call, not the implementer's.
