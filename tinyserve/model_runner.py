"""Turns scheduler batches into tensors, runs the model, returns token ids.

Owns the GPU (or CPU) state: model weights, the paged KV pool, the DFlash
draft and its KV pool, and the CUDA graphs used for decode.

Three batch kinds (see scheduler.py):
  PREFILL  every uncached prompt token of each sequence, varlen.
  DECODE   one new token per sequence (CUDA-graph replay when possible).
  SPEC     DFlash/oracle/random draft of 15 tokens, then one varlen target
           forward that verifies [last token, d_1..d_15] (PLAN.md 4.3).

The KV pool has ONE extra block at the end (`self.dummy_block`) that the
block manager never hands out; padded rows of a CUDA-graph replay write to
and read from it, so they can never clobber real KV.
"""

from __future__ import annotations

import time

import torch

from tinyserve.attention import AttnMetadata, set_attn_metadata
from tinyserve.config import EngineConfig
from tinyserve.loader import load_weights
from tinyserve.models.qwen3 import Qwen3ForCausalLM
from tinyserve.sampling import sample
from tinyserve.scheduler import Batch, BatchKind
from tinyserve.sequence import Sequence
from tinyserve.spec.verify import greedy_accept

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32, "float64": torch.float64}


class ModelRunner:
    def __init__(self, cfg: EngineConfig, hf_config, model_path, draft=None, proposer=None):
        """`draft`: (draft_hf_config, draft_path) for DFlash; `proposer`: an
        explicit Proposer object (tests), else built from cfg.spec_method."""
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = DTYPES[cfg.dtype]
        self.hf_config = hf_config
        self.block_size = cfg.block_size
        torch.manual_seed(cfg.seed)
        self.generator = torch.Generator().manual_seed(cfg.seed)  # CPU generator for sampling
        self.seq_generators: dict[int, torch.Generator] = {}
        self.stats = {"forward_s": 0.0, "steps": 0}
        self.record_gaps = False  # tests: store top1-top2 logit gaps per emitted token

        max_positions = cfg.max_model_len + 2 * cfg.spec_block_size + 8
        aux_ids = None
        if cfg.spec_method == "dflash":
            from tinyserve.spec.dflash import draft_target_layer_ids

            aux_ids = draft_target_layer_ids(draft[0])
        with torch.device(self.device):
            prev = torch.get_default_dtype()
            torch.set_default_dtype(self.dtype)
            try:
                self.model = Qwen3ForCausalLM(hf_config, aux_ids, cfg.attn_backend, cfg.use_triton_store, max_positions, self.dtype)
                self.draft_model = None
                if cfg.spec_method == "dflash":
                    from tinyserve.spec.dflash import DFlashDraftModel

                    if draft[0].block_size != cfg.spec_block_size:
                        raise ValueError(f"draft block_size {draft[0].block_size} != spec_block_size {cfg.spec_block_size}")
                    self.draft_model = DFlashDraftModel(draft[0], self.model, cfg.attn_backend, cfg.use_triton_store, max_positions, self.dtype)
            finally:
                torch.set_default_dtype(prev)
        if model_path is not None:
            load_weights(self.model, model_path)
        if self.draft_model is not None and draft[1] is not None:
            load_weights(self.draft_model, draft[1])
        self.model.eval()
        self.layers = self.model.attention_layers()
        self.draft_layers = self.draft_model.attention_layers() if self.draft_model is not None else []

        self.proposer = proposer
        if self.proposer is None and cfg.spec_method:
            from tinyserve.spec.proposer import make_proposer

            self.proposer = make_proposer(cfg, self)

        self.num_kv_blocks = self._allocate_kv_cache()
        self.dummy_block = self.num_kv_blocks  # the extra block, never allocated
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.spec_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        if not cfg.enforce_eager and self.device.type == "cuda":
            if cfg.spec_method:
                self._capture_spec_graphs()  # M9(c): the target VERIFY forward; the draft stays eager
            else:
                self._capture_cuda_graphs()

    # ------------------------------------------------------------------ KV pool
    def _kv_bytes_per_block(self) -> int:
        n_layers = len(self.layers) + len(self.draft_layers)
        a = self.layers[0]
        return 2 * n_layers * self.block_size * a.num_kv_heads * a.head_dim * self.dtype.itemsize

    @torch.inference_mode()
    def _allocate_kv_cache(self) -> int:
        cfg = self.cfg
        if cfg.num_kv_blocks is not None:
            n = cfg.num_kv_blocks
        elif self.device.type == "cuda":
            n = self._profile_num_blocks()
        else:  # CPU: enough for every sequence at full length
            n = cfg.max_num_seqs * cfg.max_blocks_per_seq
        a = self.layers[0]
        shape = (n + 1, self.block_size, a.num_kv_heads, a.head_dim)  # +1 = dummy block
        self.kv_cache = torch.zeros(2, len(self.layers), *shape, dtype=self.dtype, device=self.device)
        for i, layer in enumerate(self.layers):
            layer.k_cache, layer.v_cache = self.kv_cache[0, i], self.kv_cache[1, i]
        if self.draft_layers:
            self.draft_kv_cache = torch.zeros(2, len(self.draft_layers), *shape, dtype=self.dtype, device=self.device)
            for i, layer in enumerate(self.draft_layers):
                layer.k_cache, layer.v_cache = self.draft_kv_cache[0, i], self.draft_kv_cache[1, i]
        return n

    def _profile_num_blocks(self) -> int:
        """nano-vllm style: run the biggest forward we will ever run (a full
        token budget of prefill) with no KV cache, record the peak activation
        memory, and give the rest of `gpu_memory_utilization` to KV blocks."""
        cfg = self.cfg
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        n_tok = cfg.max_num_batched_tokens
        lens = [cfg.max_model_len] * (n_tok // cfg.max_model_len)
        if n_tok % cfg.max_model_len:
            lens.append(n_tok % cfg.max_model_len)
        ids = torch.zeros(n_tok, dtype=torch.long, device=self.device)
        pos = torch.cat([torch.arange(l, device=self.device) for l in lens])
        cu = torch.tensor([0] + list(_cumsum(lens)), dtype=torch.int32, device=self.device)
        set_attn_metadata(AttnMetadata(is_varlen=True, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=max(lens),
                                       max_seqlen_k=max(lens), q_lens=lens, k_lens=lens))
        hidden, aux = self.model(ids, pos)
        rows = max(cfg.max_num_seqs * (cfg.spec_block_size if cfg.spec_method else 1), len(lens))
        self.model.compute_logits(hidden[: min(rows, n_tok)]).argmax(-1)
        set_attn_metadata(None)
        del hidden, aux
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        stats = torch.cuda.memory_stats()
        used = total - free
        peak, current = stats["allocated_bytes.all.peak"], stats["allocated_bytes.all.current"]
        budget = total * cfg.gpu_memory_utilization - used - (peak - current)
        if cfg.spec_method == "dflash":
            # The draft's context projection runs on up to a full token budget
            # of aux hidden (5*H wide) and pending_hidden is kept per sequence.
            h = self.hf_config.hidden_size
            budget -= 3 * n_tok * len(self.model.aux_layer_ids) * h * self.dtype.itemsize
        n = int(budget // self._kv_bytes_per_block())
        if n < 1:
            raise RuntimeError("not enough GPU memory for the KV cache")
        return n - 1  # one block is the dummy

    # ------------------------------------------------------------- tensor prep
    def _slots(self, seq: Sequence, start: int, end: int) -> list[int]:
        """Flat cache slots for positions [start, end) of seq."""
        bs, bt = self.block_size, seq.block_table
        out: list[int] = []
        p = start
        while p < end:
            b, off = divmod(p, bs)
            stop = min(end, (b + 1) * bs)
            base = bt[b] * bs
            out.extend(range(base + off, base + off + (stop - p)))
            p = stop
        return out

    def _block_tables(self, seqs: list[Sequence], width: int | None = None) -> torch.Tensor:
        width = width or max(len(s.block_table) for s in seqs)
        rows = [s.block_table + [self.dummy_block] * (width - len(s.block_table)) for s in seqs]
        return torch.tensor(rows, dtype=torch.int32, device=self.device)

    def _t(self, x, dtype=torch.long) -> torch.Tensor:
        return torch.tensor(x, dtype=dtype, device=self.device)

    def _varlen_meta(self, seqs, q_lens, k_lens, slots, causal=True, paged=True) -> AttnMetadata:
        return AttnMetadata(
            is_varlen=True, causal=causal,
            cu_seqlens_q=self._t([0, *_cumsum(q_lens)], torch.int32),
            cu_seqlens_k=self._t([0, *_cumsum(k_lens)], torch.int32),
            max_seqlen_q=max(q_lens), max_seqlen_k=max(k_lens),
            slot_mapping=self._t(slots),
            block_tables=self._block_tables(seqs) if paged else None,
            q_lens=list(q_lens), k_lens=list(k_lens),
        )

    # ------------------------------------------------------------------- run
    @torch.inference_mode()
    def run(self, batch: Batch) -> list[list[int]]:
        t0 = time.perf_counter()
        if batch.kind == BatchKind.PREFILL:
            out = self.run_prefill(batch.seqs)
        elif batch.kind == BatchKind.DECODE:
            out = self.run_decode(batch.seqs)
        else:
            out = self.run_spec(batch.seqs)
        set_attn_metadata(None)
        self.stats["forward_s"] += time.perf_counter() - t0
        self.stats["steps"] += 1
        return out

    def _record_gaps(self, logits: torch.Tensor, seqs: list[Sequence], rows_per_seq: int = 1) -> None:
        top2 = logits.float().topk(2, dim=-1).values
        gaps = (top2[:, 0] - top2[:, 1]).tolist()
        for i, s in enumerate(seqs):
            s.logit_gaps.extend(gaps[i * rows_per_seq : (i + 1) * rows_per_seq])

    def _sample(self, logits: torch.Tensor, seqs: list[Sequence]) -> list[int]:
        if self.record_gaps:
            self._record_gaps(logits, seqs)
        params = [s.params for s in seqs]
        if all(p.greedy for p in params):
            return logits.argmax(-1).tolist()
        gens = []
        for s in seqs:
            if s.params.seed is not None:
                if s.seq_id not in self.seq_generators:
                    self.seq_generators[s.seq_id] = torch.Generator().manual_seed(s.params.seed)
                gens.append(self.seq_generators[s.seq_id])
            else:
                gens.append(self.generator)
        return sample(logits, params, gens)

    def release(self, seq: Sequence) -> None:
        self.seq_generators.pop(seq.seq_id, None)

    def run_prefill(self, seqs: list[Sequence]) -> list[list[int]]:
        ids, pos, slots, q_lens, k_lens = [], [], [], [], []
        for s in seqs:
            c, n = s.num_cached_tokens, s.num_tokens
            ids += s.token_ids[c:n]
            pos += range(c, n)
            slots += self._slots(s, c, n)
            q_lens.append(n - c)
            k_lens.append(n)
        # Read K/V from the paged cache only if some sequence has a cached prefix.
        paged = any(s.num_cached_tokens > 0 for s in seqs)
        set_attn_metadata(self._varlen_meta(seqs, q_lens, k_lens, slots, paged=paged))
        hidden, aux = self.model(self._t(ids), self._t(pos))
        last = self._t(_cumsum(q_lens)) - 1
        logits = self.model.compute_logits(hidden[last])
        if self.draft_model is not None:
            # DFlash: the draft has not seen the computed positions [c, n-1] yet.
            q0 = 0
            for s, ql in zip(seqs, q_lens):
                s.pending_hidden = aux[q0 : q0 + ql].clone()
                s.draft_ctx_len = s.num_cached_tokens
                q0 += ql
        return [[t] for t in self._sample(logits, seqs)]

    def run_decode(self, seqs: list[Sequence]) -> list[list[int]]:
        n = len(seqs)
        graph_bs = next((b for b in sorted(self.graphs) if b >= n), None)
        if graph_bs is not None:
            hidden = self._replay_graph(seqs, graph_bs)
        else:
            k_lens = [s.num_tokens for s in seqs]
            set_attn_metadata(AttnMetadata(
                is_varlen=False,
                slot_mapping=self._t([self._slots(s, s.num_tokens - 1, s.num_tokens)[0] for s in seqs]),
                context_lens=self._t(k_lens, torch.int32),
                block_tables=self._block_tables(seqs), k_lens=k_lens,
            ))
            hidden, _ = self.model(self._t([s.last_token for s in seqs]), self._t([s.num_tokens - 1 for s in seqs]))
        logits = self.model.compute_logits(hidden)
        return [[t] for t in self._sample(logits, seqs)]

    def run_spec(self, seqs: list[Sequence]) -> list[list[int]]:
        """One speculative step (PLAN.md 4.3): draft K tokens, verify all of
        them with one target forward, keep the longest agreeing prefix + 1."""
        blk = self.cfg.spec_block_size
        drafts = self.proposer.propose(seqs, self)  # [B, blk-1] (also advances draft_ctx_len)
        assert drafts.shape == (len(seqs), blk - 1)
        ids = torch.cat([self._t([[s.last_token] for s in seqs]), drafts.to(self.device)], dim=1).flatten()
        pos, slots, k_lens = [], [], []
        for s in seqs:
            n = s.num_tokens
            pos += range(n - 1, n - 1 + blk)
            slots += self._slots(s, n - 1, n - 1 + blk)
            k_lens.append(n - 1 + blk)
        graph_bs = next((b for b in sorted(self.spec_graphs) if b >= len(seqs)), None)
        if graph_bs is not None:
            hidden, aux = self._replay_spec_graph(seqs, ids, pos, slots, k_lens, graph_bs)
        else:
            set_attn_metadata(self._varlen_meta(seqs, [blk] * len(seqs), k_lens, slots, paged=True))
            hidden, aux = self.model(ids, self._t(pos))
        logits = self.model.compute_logits(hidden)
        target_argmax = logits.argmax(-1).view(len(seqs), blk)
        num_acc, new_tokens = greedy_accept(drafts.to(self.device), target_argmax)
        if self.record_gaps:
            top2 = logits.float().topk(2, dim=-1).values
            gaps = (top2[:, 0] - top2[:, 1]).view(len(seqs), blk).tolist()
        for b, (s, a) in enumerate(zip(seqs, num_acc.tolist())):
            s.spec_steps += 1
            s.spec_accepted += a
            s.spec_history.append(a + 1)
            if self.record_gaps:
                s.logit_gaps.extend(gaps[b][: a + 1])
            if self.draft_model is not None:
                # Target aux hidden for the a+1 positions [n-1, n-1+a] that are now committed.
                s.pending_hidden = aux[b * blk : b * blk + a + 1].clone()
        return new_tokens

    # ------------------------------------------------------------ CUDA graphs
    @torch.inference_mode()
    def _capture_cuda_graphs(self) -> None:
        """Capture one decode graph per batch size. Replaying a graph launches
        the whole forward (hundreds of kernels) with one CPU call, which is
        what makes small-batch decode fast: otherwise the GPU idles while
        Python launches kernels one by one."""
        cfg = self.cfg
        sizes = sorted(b for b in cfg.cuda_graph_batch_sizes if b <= cfg.max_num_seqs)
        if not sizes:
            return
        max_bs, width = sizes[-1], cfg.max_blocks_per_seq
        dummy_slot = self.dummy_block * self.block_size
        h = self.hf_config.hidden_size
        self.g_input_ids = torch.zeros(max_bs, dtype=torch.long, device=self.device)
        self.g_positions = torch.zeros(max_bs, dtype=torch.long, device=self.device)
        self.g_slots = torch.full((max_bs,), dummy_slot, dtype=torch.long, device=self.device)
        self.g_context_lens = torch.ones(max_bs, dtype=torch.int32, device=self.device)
        self.g_block_tables = torch.full((max_bs, width), self.dummy_block, dtype=torch.int32, device=self.device)
        self.g_hidden = torch.zeros(max_bs, h, dtype=self.dtype, device=self.device)
        pool = None
        for bs in reversed(sizes):  # largest first so smaller graphs reuse its memory pool
            meta = AttnMetadata(is_varlen=False, slot_mapping=self.g_slots[:bs], context_lens=self.g_context_lens[:bs],
                                block_tables=self.g_block_tables[:bs])
            set_attn_metadata(meta)
            self.g_hidden[:bs] = self.model(self.g_input_ids[:bs], self.g_positions[:bs])[0]  # warmup (Triton JIT)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool):
                self.g_hidden[:bs] = self.model(self.g_input_ids[:bs], self.g_positions[:bs])[0]
            if pool is None:
                pool = g.pool()
            self.graphs[bs] = g
            torch.cuda.synchronize()
        set_attn_metadata(None)

    @torch.inference_mode()
    def _capture_spec_graphs(self) -> None:
        """M9(c): capture the spec VERIFY forward (blk tokens per sequence)
        per batch size. Its shape is static: cu_seqlens_q is always
        0, blk, 2*blk, ...; only the VALUES of cu_seqlens_k, slots, positions
        and block tables change, and those live in static buffers.
        max_seqlen_k is fixed at an upper bound (the kernel masks by the real
        cu_seqlens_k). Padded rows attend to / write into the dummy block."""
        cfg = self.cfg
        sizes = sorted(b for b in cfg.cuda_graph_batch_sizes if b <= cfg.max_num_seqs)
        if not sizes:
            return
        blk, max_bs, width = cfg.spec_block_size, sizes[-1], cfg.max_blocks_per_seq
        n_max = max_bs * blk
        dummy_slot = self.dummy_block * self.block_size
        dev = self.device
        self.v_ids = torch.zeros(n_max, dtype=torch.long, device=dev)
        self.v_pos = torch.arange(blk, device=dev).repeat(max_bs)
        self.v_slots = (dummy_slot + torch.arange(blk, device=dev)).repeat(max_bs)
        self.v_cu_q = torch.arange(max_bs + 1, dtype=torch.int32, device=dev) * blk
        self.v_cu_k = self.v_cu_q.clone()
        self.v_bt = torch.full((max_bs, width), self.dummy_block, dtype=torch.int32, device=dev)
        self.v_hidden = torch.zeros(n_max, self.hf_config.hidden_size, dtype=self.dtype, device=dev)
        n_aux = len(self.model.aux_layer_ids)
        self.v_aux = torch.zeros(n_max, n_aux * self.hf_config.hidden_size, dtype=self.dtype, device=dev) if n_aux else None
        self.v_max_k = cfg.max_model_len + blk
        pool = None
        for bs in reversed(sizes):
            n = bs * blk
            set_attn_metadata(AttnMetadata(
                is_varlen=True, causal=True, cu_seqlens_q=self.v_cu_q[: bs + 1], cu_seqlens_k=self.v_cu_k[: bs + 1],
                max_seqlen_q=blk, max_seqlen_k=self.v_max_k, slot_mapping=self.v_slots[:n], block_tables=self.v_bt[:bs]))

            def fwd():
                h, a = self.model(self.v_ids[:n], self.v_pos[:n])
                self.v_hidden[:n] = h
                if a is not None:
                    self.v_aux[:n] = a

            fwd()  # warmup (Triton JIT, allocator)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool):
                fwd()
            if pool is None:
                pool = g.pool()
            self.spec_graphs[bs] = g
            torch.cuda.synchronize()
        set_attn_metadata(None)

    def _replay_spec_graph(self, seqs, ids, pos, slots, k_lens, graph_bs):
        blk, n_real, n = self.cfg.spec_block_size, len(seqs) * self.cfg.spec_block_size, graph_bs * self.cfg.spec_block_size
        pad = graph_bs - len(seqs)
        dummy_slot = self.dummy_block * self.block_size
        self.v_ids[:n_real] = ids
        self.v_pos[:n] = self._t(pos + list(range(blk)) * pad)
        self.v_slots[:n] = self._t(slots + list(range(dummy_slot, dummy_slot + blk)) * pad)
        self.v_cu_k[: graph_bs + 1] = self._t([0, *_cumsum(k_lens + [blk] * pad)], torch.int32)
        bt = self._block_tables(seqs, self.v_bt.shape[1])
        self.v_bt[: len(seqs)] = bt
        self.v_bt[len(seqs) : graph_bs] = self.dummy_block
        self.spec_graphs[graph_bs].replay()
        return self.v_hidden[:n_real], (self.v_aux[:n_real] if self.v_aux is not None else None)

    def _replay_graph(self, seqs: list[Sequence], graph_bs: int) -> torch.Tensor:
        n = len(seqs)
        dummy_slot = self.dummy_block * self.block_size
        self.g_input_ids[:n] = self._t([s.last_token for s in seqs])
        self.g_positions[:n] = self._t([s.num_tokens - 1 for s in seqs])
        slots = [self._slots(s, s.num_tokens - 1, s.num_tokens)[0] for s in seqs]
        self.g_slots[:graph_bs] = self._t(slots + [dummy_slot] * (graph_bs - n))
        self.g_context_lens[:graph_bs] = self._t([s.num_tokens for s in seqs] + [1] * (graph_bs - n), torch.int32)
        self.g_block_tables[:graph_bs] = self._block_tables(seqs, self.g_block_tables.shape[1]) if n == graph_bs else \
            torch.cat([self._block_tables(seqs, self.g_block_tables.shape[1]),
                       torch.full((graph_bs - n, self.g_block_tables.shape[1]), self.dummy_block, dtype=torch.int32, device=self.device)])
        self.graphs[graph_bs].replay()
        return self.g_hidden[:n]


def _cumsum(xs) -> list[int]:
    out, total = [], 0
    for x in xs:
        total += x
        out.append(total)
    return out
