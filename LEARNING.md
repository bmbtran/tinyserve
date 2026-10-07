# LEARNING.md: LLM inference, explained through tinyserve

Plain-English notes for interviews. Each section: **problem → intuition → numbers → where it lives in
tinyserve → likely questions with short answers.** Measured numbers come from `results/` (see RESULTS.md).

Hardware used throughout: **NVIDIA L4** (24 GB GDDR6, **300 GB/s**, **121 dense BF16 TFLOPS**; NVIDIA's
headline 242 TFLOPS is with 2:4 sparsity).

---

## 1. Prefill vs decode, and why decode is memory-bound

**Problem.** Generating text has two phases. *Prefill* runs the whole prompt through the model at once.
*Decode* produces one token per sequence per step, and every step must read every weight.

**Intuition.** Prefill is like a bus carrying hundreds of passengers per trip; decode at batch 1 is the
same bus carrying one passenger. The trip (reading the weights from memory) costs the same.

**Numbers.** A matmul with a weight of P parameters does ~2·P FLOPs per token and reads 2·P bytes (bf16).
- Arithmetic intensity of decode at batch B ≈ 2·P·B FLOPs / 2·P bytes = **B FLOP/byte**.
- L4 ridge point = 121e12 / 300e9 ≈ **403 FLOP/byte**. Below that, memory bandwidth is the limit.
- So decode stays memory-bound until batch ≈ 400. Prefill of a 512-token prompt has intensity ~512: compute-bound.
- Lower bound for one decode step = weight bytes / bandwidth: Qwen3-0.6B 1.2 GB → **4.0 ms**; Qwen3-4B 8.0 GB → **27 ms (≤ 37 tok/s at batch 1)**.
- Measured: Qwen3-0.6B decode with CUDA graphs = 7.75 ms/token at bs=1 (`results/verify/m5.json`), ~2× the bound
  (the gap is lm_head over a 151,936 vocab, attention, small kernels, norms).

**Roofline sketch (L4).**
```
 TFLOPS
 121 |                    ______________________ compute roof
     |                  /
     |                /   <- prefill (intensity ~ prompt length) sits here
     |              /
     |            /  slope = 300 GB/s
     |  decode  /
     |  (B=1) /
     +------403----------------------------> FLOP/byte
```

**In tinyserve.** `scheduler.py: Scheduler.schedule` (prefill first, then decode);
`model_runner.py: run_prefill / run_decode`.

**Questions.**
- *Why is decode memory-bound?* Each step reads all weights to produce one token per sequence; intensity ≈ batch size, far under the GPU's ~400 FLOP/byte ridge.
- *How do you make decode faster?* Batch more sequences per weight read (continuous batching), verify several tokens per read (speculative decoding), shrink the bytes (quantization), cut CPU overhead (CUDA graphs).
- *Why does TTFT grow with prompt length but ITL not much?* TTFT is prefill (compute ∝ prompt tokens); ITL is a decode step (dominated by weight reads, nearly independent of context until attention gets long).

## 2. The KV cache

**Problem.** Attention at step t needs keys and values of all previous tokens. Recomputing them is O(n²).

**Intuition.** Keep the K and V vectors of every past token: that is the KV cache. Each new token adds one row per layer.

**Numbers.** bytes/token = 2 (K and V) × layers × kv_heads × head_dim × bytes.
- Qwen3-0.6B: 2 × 28 × 8 × 128 × 2 B = **112 KiB/token** → a 4,096-token sequence = 448 MiB.
- Qwen3-4B: 2 × 36 × 8 × 128 × 2 B = **144 KiB/token** → 576 MiB per 4,096-token sequence.
- GQA (grouped-query attention) is why this is small: Qwen3-4B has 32 query heads but only 8 KV heads, so the cache is 4× smaller than with full multi-head attention.
- On the L4 at 85% memory, tinyserve got 613 blocks × 256 tokens = **157k tokens** of KV for Qwen3-0.6B (`results/verify/m3.log`).

**In tinyserve.** `model_runner.py: ModelRunner._allocate_kv_cache / _profile_num_blocks` (one tensor
`[2, layers, blocks+1, block_size, kv_heads, head_dim]`, sized after a profiling forward).

**Questions.**
- *What limits the max batch size?* KV memory: (free memory after weights and activations) / (KV bytes per token × tokens per sequence).
- *What does GQA trade?* Fewer KV heads → smaller cache and less bandwidth, at a small quality cost.
- *Why is the KV cache memory-bound to read?* Each decode step reads the whole cache of every sequence once; at long contexts that rivals the weights.

## 3. PagedAttention

**Problem.** Reserving max_model_len contiguous slots per request wastes most of the memory (internal
fragmentation) and makes sharing impossible.

**Intuition.** Virtual memory for KV. The cache is a pool of fixed-size *blocks*; each sequence has a
*block table* (its page table). Position p lives at `slot = block_table[p // B] * B + p % B`.

**Numbers.** Waste is at most one partial block per sequence. With block size 256 that is up to 255
tokens (~28 MiB for Qwen3-0.6B) per sequence; vLLM uses 16 for this reason, but FlashAttention-2's paged
kernels require multiples of 256, so tinyserve's GPU path uses 256 (the CPU torch backend accepts any size).

**In tinyserve.** `block_manager.py: BlockManager.allocate / ensure_slots / deallocate`;
`model_runner.py: ModelRunner._slots` (slot mapping); `attention/triton_store.py` (writes K/V into slots);
`attention/backend_flash.py` (`block_table=` argument).

**Questions.**
- *What is a slot mapping?* The flat cache index where each new token's K/V is written this step.
- *How do ref counts enable sharing?* A block used by two sequences has ref_count 2; it returns to the free list only when both release it. Writes only ever go to unshared blocks (shared ones are full and immutable), so no copy-on-write is needed for prefix sharing.
- *Why not block size 1?* Per-block metadata and gather overhead, and kernels want contiguous tiles.

## 4. Continuous batching and preemption

**Problem.** Static batching waits for the longest request in a batch; short requests and new arrivals idle (head-of-line blocking).

**Intuition (Orca).** Schedule per *iteration*, not per request: after every step, finished sequences
leave and waiting ones join.

**Numbers.** Offline Qwen3-0.6B, 256 × (512 in / 256 out), eager: **2,004 output tok/s** (`results/verify/m3.json`).

**Preemption.** When the KV pool runs out, tinyserve frees the most recently admitted sequence and
re-queues it at the front with prompt + generated tokens as its new prompt (*recompute*). The
alternative, *swapping* KV to CPU memory, costs PCIe bandwidth instead of compute. Recompute is simpler and
often cheaper for short contexts; with prefix caching, many of the freed blocks are found again by hash.

**In tinyserve.** `scheduler.py: Scheduler.schedule / _schedule_prefill / _preempt`.
Tested exact in float64: `tests/cpu/test_engine_batching.py` (batched == preempted == random arrival == solo).

**Questions.**
- *What happens when the KV cache is full?* New prefills wait; a running sequence that cannot get its next slot triggers preemption of the newest sequence (recompute later).
- *Recompute vs swap?* Recompute burns GPU FLOPs (cheap for prefill, which is compute-efficient); swap burns PCIe (~25 GB/s) and CPU memory. vLLM V1 defaults to recompute.
- *Why prefill-first?* New requests get their first token fast (TTFT), at the cost of a decode hiccup (ITL spike) for the running ones. Chunked prefill (Sarathi) fixes the hiccup.

## 5. Prefix caching

**Problem.** Many requests share a prefix (system prompt, few-shot examples, chat history). Recomputing it every time wastes prefill compute.

**Intuition.** Name each full block by a hash of (parent block's hash, its tokens). A chain hash means a
hit on block i implies the whole prefix 0..i matched. Lookup = walk the chain until the first miss.

**Hash chain (vLLM APC, tinyserve) vs radix tree (SGLang RadixAttention).**
- Hash chain: a flat dict, block granularity, trivial to implement; eviction by an LRU-ish free list.
- Radix tree: token granularity, natural longest-prefix match and LRU over tree nodes, handles branching conversations elegantly; more code.
- Block size 256 hurts hit granularity: a 2,100-token shared prefix only reuses 2,048.

**Numbers.** 2,048-token shared prefix + 64 unique tokens: hit rate 0.943, TTFT p50 **553.7 → 56.1 ms (9.9×)** (`results/verify/m4.json`).

**In tinyserve.** `block_manager.py: chain_hash / _match_prefix / commit_full_blocks`. Hashes are
registered only after KV is computed (and, with DFlash, after the *draft* KV is too).

**Questions.**
- *Why must at least one prompt token be recomputed?* We need logits for the last position to sample the first token.
- *How does eviction work?* Freed blocks keep their hash; they are recycled oldest-first, and the hash is dropped when a block is reused.
- *How does prefix caching interact with spec decoding in your design?* A block is cached only when both target KV and draft KV are valid for it (`min(target_valid, draft_ctx_len)`), so a hit gives the draft its context for free too.

## 6. FlashAttention

**Problem.** Naive attention materializes an n×n score matrix in GPU memory: O(n²) memory traffic.

**Intuition.** Tile Q, K, V into on-chip SRAM blocks and compute softmax *online* (keep a running max and
running sum, rescale as new tiles arrive), so the n×n matrix never touches HBM. Same math, IO-aware.

**Variants used.** `flash_attn_varlen_func` packs sequences of different lengths with `cu_seqlens`
(no padding), and with `block_table` reads K/V from the paged cache (prefix-cached prefill, spec verify,
the non-causal DFlash draft). `flash_attn_with_kvcache` is the decode kernel (one query per sequence).

**In tinyserve.** `attention/backend_flash.py`; the obviously-correct oracle is `attention/backend_torch.py`.

**Questions.**
- *What is online softmax?* softmax(x) = exp(x − m)/Σexp(x − m); if a later tile has a bigger max m', rescale the old sum by exp(m − m').
- *Why varlen?* Padding a batch to the longest prompt wastes compute; cu_seqlens describes the packed batch.
- *Causal mask when q_len < k_len?* Bottom-right aligned: query i sits at absolute position k_len − q_len + i.

## 7. CUDA graphs

**Problem.** A decode step of a small model is hundreds of tiny kernels; Python + PyTorch dispatch take
~10-20 µs each, so the GPU waits on the CPU.

**Intuition.** Record the whole forward once as a graph; replay it with one CPU call. Shapes must be
static, so capture one graph per batch size bucket and pad the real batch up to the next bucket.

**Numbers.** Qwen3-0.6B decode: **39.1 ms eager → 7.75 ms with graphs at bs=1 (5.0×)**; 41.0 → 10.2 ms at
bs=32; the gain shrinks at bs=128 (2.4×) as the GPU work grows (`results/verify/m5.json`).

**In tinyserve.** `model_runner.py: _capture_cuda_graphs / _replay_graph`. Padded rows point at a reserved
dummy KV block, so they can never overwrite real KV.

**Questions.**
- *Why not graph prefill?* Prefill shapes vary a lot and it is compute-bound anyway, so launch overhead is a small fraction.
- *What breaks graphs?* CPU syncs (`.item()`, data-dependent shapes like boolean masks), allocations, changing tensor addresses. Inputs must be copied into static buffers.
- *Why share a memory pool?* Each graph's private allocations would otherwise add up; captured largest-first, smaller graphs reuse the pool.

## 8. Speculative decoding

**Problem.** Decode is memory-bound, so verifying 16 tokens in one forward costs about the same as producing 1.

**Intuition.** A cheap *draft* guesses k tokens; the target scores all of them in one forward; keep the
longest prefix where the draft agreed with the target's own greedy choice, plus one "bonus" token from the
target. Every emitted token is exactly what greedy target decoding would produce: **lossless**.

**Numbers.** Expected tokens per step = τ (acceptance length incl. bonus). Speedup ≈ τ × (target step time)
/ (draft time + verify time). DFlash on Qwen3-4B: τ ≈ 5.5-6 on GSM8K/HumanEval (`results/verify/m7.json`).

**Temperature > 0.** Rejection sampling (Leviathan et al.; Chen et al.): accept draft token x with
probability min(1, p(x)/q(x)), else resample from max(0, p − q). It preserves the target distribution
exactly. tinyserve implements greedy only (a non-goal).

**In tinyserve.** `spec/verify.py: greedy_accept`; `model_runner.py: run_spec`; `spec/proposer.py`.

**Questions.**
- *Why is greedy verification lossless?* Each kept token equals the target's argmax given the same prefix, so the sequence is identical to greedy decoding.
- *Why does spec decoding help less at high batch size?* At large batch the target step becomes compute-bound: verifying 16 tokens per sequence costs ~16× a normal decode step, and rejected tokens are wasted compute.
- *How would you pick the block size?* Larger blocks raise τ with diminishing returns but cost more verify compute; pick the k that maximizes τ / step-cost at your target concurrency (DFlash was trained with 16).
- *How do you test it when the draft could be wrong?* Correctness never depends on the draft. Oracle (τ = 16) and random (τ ≈ 1) proposers test the plumbing; τ parity with the reference tests the draft.

## 9. EAGLE → EAGLE-3 → DFlash

- **Medusa:** extra heads on the target predict t+2, t+3, ...: cheap but weak beyond 2-3 tokens.
- **EAGLE:** an autoregressive 1-layer draft that runs on the target's *features* (hidden states) rather than tokens, so it inherits most of the target's understanding.
- **EAGLE-3:** fuses features from several target layers and trains with "training-time test" (feeding its own predictions back during training), lifting τ to ~3-4 on 4B-class models.
- **DFlash:** drafts the whole block in **one non-causal forward** (block diffusion: [t, MASK × 15] denoised at once). Target features from 5 layers are projected by `fc` and *injected into the draft's KV cache* as context. No sequential draft loop, so drafting 15 tokens costs one small forward, and τ reaches ~6.5 on GSM8K in the paper.
- **In tinyserve:** `spec/dflash.py: DFlashDraftModel`, `spec/proposer.py: DFlashProposer`; the draft KV shares the target's block tables, so paging, preemption and prefix caching just work.

**Questions.**
- *Why parallel block drafting beats autoregressive drafting:* draft cost is one forward regardless of k, and the draft attends bidirectionally within the block.
- *What does "KV injection of target features" mean?* Context positions contribute only keys/values (from projected target hidden states), never queries; they are cached like normal KV.

## 10. Serving metrics

- **TTFT** (time to first token): queueing + prefill. Users feel it as "is it thinking?".
- **ITL** (inter-token latency): gap between streamed chunks. With spec decoding a chunk can carry several tokens.
- **TPOT** (time per output token) = (e2e − TTFT)/(output tokens − 1).
- **Throughput:** output tokens/s across all requests. **Goodput:** throughput of requests that meet the SLO (e.g. TTFT < 1 s and TPOT < 50 ms).
- The tradeoff: raising concurrency raises throughput (bigger batches per weight read) until compute or KV memory saturates, while TTFT and TPOT degrade.

## 11. Disaggregated prefill/decode and chunked prefill (concepts)

- **Chunked prefill (Sarathi-Serve):** split long prompts into chunks and mix them with decode tokens in one step, so prefills do not stall decodes (bounded ITL).
- **Disaggregation (DistServe, Mooncake):** run prefill and decode on different GPUs, each tuned for its phase, and ship the KV cache between them over fast interconnect. Wins appear at scale under strict SLOs; not reproducible on one L4.

## 12. How serverless GPU platforms serve models (Modal / Nebius / Baseten)

- **Cold start** = container start + weight load + (CUDA graph capture / compile). tinyserve's server became healthy ~18 s after launch for Qwen3-0.6B with weights already in a Modal Volume (`results/verify/m6.log`).
- **Weight caching:** volumes or local NVMe snapshots avoid re-downloading GBs per container.
- **Autoscaling:** scale-to-zero saves money but adds cold starts; keep-warm (min containers) trades money for latency.
- **Cost per 1M tokens** = $/hour ÷ (tokens/s × 3,600) × 1e6. Worked example with this project's numbers: see README "Cost".

## 13. Interview question bank

1. Why is decode memory-bound? → §1.
2. Estimate the max decode tokens/s of an 8B bf16 model on an H100 (3.35 TB/s) at batch 1. → 16 GB / 3.35 TB/s ≈ 4.8 ms → ~200 tok/s upper bound.
3. What happens when the KV cache is full? → §4.
4. Recompute vs swap preemption? → §4.
5. How big is the KV cache of Qwen3-4B at 4k context? → 576 MiB per sequence (§2).
6. What does GQA buy you? → 4× smaller KV for Qwen3-4B (§2).
7. Walk me through a request's life in your engine. → HTTP handler tokenizes → AsyncEngine queue → engine thread `add_request` → scheduler admits (prefix-cache lookup, block allocation) → PREFILL forward → sample → DECODE/SPEC steps with continuous batching → per-step SSE chunks, incremental detokenization → stop condition → blocks freed (hashes kept for reuse).
8. What is a block table and a slot mapping? → §3.
9. Why does FA2 paged KV need block size 256, and what does that cost you? → kernel constraint; coarser prefix-cache granularity and more partial-block waste.
10. How does prefix caching decide a hit? → chained hash (§5).
11. Hash chain vs radix tree? → §5.
12. Why keep at least one prompt token uncached? → §5.
13. How does prefix caching interact with spec decoding in your design? → §5/§9: commit only when draft KV is valid.
14. Why does continuous batching beat static batching? → §4.
15. Prefill-first scheduling: pros and cons? → §4.
16. What is chunked prefill for? → §11.
17. What are CUDA graphs and why do they help decode? → §7.
18. What breaks CUDA-graph capture? → §7.
19. How do you handle a batch size with no captured graph? → pad to the next bucket; padded rows use a dummy block.
20. Explain online softmax. → §6.
21. Why is speculative decoding lossless under greedy? → §8.
22. How does speculative sampling stay exact at temperature > 0? → rejection sampling (§8).
23. Why does spec decoding help less at high batch size? → §8.
24. How would you pick the speculative block size? → §8.
25. EAGLE vs DFlash? → §9.
26. How do you test an inference engine for correctness when bf16 is nondeterministic? → exactness in float64 on CPU with a torch reference backend; on GPU only a pre-declared near-tie rule (divergence allowed only where the reference's top-1/top-2 logit gap < 0.5); oracle/random proposers for spec.
27. Two attention kernels give different greedy outputs in bf16. Bug or not? → Check the logit gap at the first divergence and compare against a baseline (HF sdpa vs HF eager diverged on 5/8 prompts in our M2 run). If the gap is ~1 ulp, it is rounding.
28. TTFT vs TPOT: which matters for chat? for batch summarization? → chat: both (TTFT for perceived latency, TPOT for reading speed); batch jobs: throughput and cost.
29. What is goodput? → §10.
30. Why might a small engine lose to vLLM on throughput? → CPU overhead per step (Python scheduling, tensor prep), no overlap of CPU scheduling with GPU work, no fused kernels (RMSNorm, RoPE, SwiGLU), no chunked prefill, server and engine sharing one Python process (GIL).
31. How would you serve this on Modal cheaply? → weights in a Volume, `scaledown_window` short, `max_containers` capped, CUDA graphs captured at startup, one container per GPU, scale-to-zero for dev.
32. Compute the cost per 1M output tokens from a throughput number. → §12 formula.
33. What would you build next? → chunked prefill, overlap scheduling, CUDA graphs for the spec verify step, EAGLE-3 comparison, FP8 weights.
