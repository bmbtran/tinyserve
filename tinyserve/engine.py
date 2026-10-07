"""LLMEngine (step loop) and LLM (offline generate API).

    engine.step():
        batch = scheduler.schedule()            # PREFILL | DECODE | SPEC
        new_tokens = model_runner.run(batch)    # list of token lists
        finished = scheduler.postprocess(batch, new_tokens)

    llm = LLM("Qwen/Qwen3-0.6B")
    llm.generate(["Hello"], SamplingParams(max_tokens=32))
"""

from __future__ import annotations

import json
from pathlib import Path

from tinyserve.block_manager import BlockManager
from tinyserve.config import EngineConfig
from tinyserve.loader import resolve_model_path
from tinyserve.model_runner import ModelRunner
from tinyserve.sampling import SamplingParams
from tinyserve.scheduler import Batch, BatchKind, Scheduler
from tinyserve.sequence import Sequence


def _eos_ids(path: Path, hf_config) -> list[int]:
    """HF `generate` stops on generation_config.json's eos ids (Qwen3: both
    <|im_end|> and <|endoftext|>), so we do too."""
    gen = path / "generation_config.json"
    eos = None
    if gen.exists():
        eos = json.loads(gen.read_text(encoding="utf-8")).get("eos_token_id")
    if eos is None:
        eos = getattr(hf_config, "eos_token_id", None)
    if eos is None:
        return []
    return [eos] if isinstance(eos, int) else list(eos)


class LLMEngine:
    def __init__(self, cfg: EngineConfig, proposer=None, load_tokenizer: bool = True):
        from transformers import AutoConfig

        self.cfg = cfg.validate()
        self.model_path = resolve_model_path(cfg.model)
        self.hf_config = AutoConfig.from_pretrained(self.model_path)
        draft = None
        if cfg.spec_method == "dflash":
            draft_path = resolve_model_path(cfg.spec_draft_model)
            draft = (AutoConfig.from_pretrained(draft_path), draft_path)
        self.tokenizer = None
        if load_tokenizer and (self.model_path / "tokenizer_config.json").exists():
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.runner = ModelRunner(cfg, self.hf_config, self.model_path, draft, proposer)
        self.block_manager = BlockManager(self.runner.num_kv_blocks, cfg.block_size, cfg.enable_prefix_cache)
        self.eos_token_ids = _eos_ids(self.model_path, self.hf_config)
        self.scheduler = Scheduler(cfg, self.block_manager, self.eos_token_ids)
        self.counters = {"spec_steps": 0, "spec_accepted": 0, "prompt_tokens": 0, "generation_tokens": 0,
                         "finished": 0, "steps": {k.name: 0 for k in BatchKind}}

    # ------------------------------------------------------------- requests
    def encode(self, prompt: str | list[int]) -> list[int]:
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("no tokenizer loaded; pass token ids")
            return self.tokenizer.encode(prompt)
        return list(prompt)

    def apply_chat_template(self, messages: list[dict], **template_kwargs) -> list[int]:
        """Chat messages -> prompt token ids (thinking disabled by default)."""
        template_kwargs.setdefault("enable_thinking", False)
        return self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **template_kwargs)

    def add_request(self, prompt: str | list[int], params: SamplingParams | None = None) -> Sequence:
        seq = Sequence(self.encode(prompt), params or SamplingParams())
        self.scheduler.add(seq)
        self.counters["prompt_tokens"] += seq.num_prompt_tokens
        return seq

    def abort(self, seq: Sequence) -> None:
        self.scheduler.abort(seq)
        self.runner.release(seq)

    def has_work(self) -> bool:
        return self.scheduler.has_work()

    # ----------------------------------------------------------------- step
    def step(self) -> tuple[Batch | None, list[Sequence]]:
        """Run one batch. Returns (the batch that ran, sequences that finished)."""
        batch = self.scheduler.schedule()
        if batch is None:
            return None, []
        before = [s.num_tokens for s in batch.seqs]
        new_tokens = self.runner.run(batch)
        if batch.kind == BatchKind.SPEC:
            self.counters["spec_steps"] += len(batch.seqs)
            self.counters["spec_accepted"] += sum(len(t) - 1 for t in new_tokens)
        finished = self.scheduler.postprocess(batch, new_tokens)
        self.counters["steps"][batch.kind.name] += 1
        self.counters["generation_tokens"] += sum(s.num_tokens - b for s, b in zip(batch.seqs, before))
        for s in finished:
            self.runner.release(s)
        self.counters["finished"] += len(finished)
        return batch, finished

    def metrics(self) -> dict:
        bm = self.block_manager
        c = self.counters
        return {
            "running": len(self.scheduler.running),
            "waiting": len(self.scheduler.waiting),
            "kv_usage": bm.num_used_blocks / bm.num_blocks,
            "num_kv_blocks": bm.num_blocks,
            "prefix_hit_rate": bm.prefix_hit_rate,
            "prefix_query_tokens": bm.stats["prefix_query_tokens"],
            "prefix_hit_tokens": bm.stats["prefix_hit_tokens"],
            "spec_tau": (c["spec_accepted"] + c["spec_steps"]) / c["spec_steps"] if c["spec_steps"] else None,
            "spec_steps": c["spec_steps"],
            "num_preemptions": self.scheduler.num_preemptions,
            "prompt_tokens": c["prompt_tokens"],
            "generation_tokens": c["generation_tokens"],
            "finished": c["finished"],
            "steps": dict(c["steps"]),
        }

    # ------------------------------------------------------------- offline
    def generate(self, prompts: list[str | list[int]], params: SamplingParams | list[SamplingParams] | None = None) -> list[dict]:
        params = params or SamplingParams()
        plist = params if isinstance(params, list) else [params] * len(prompts)
        seqs = [self.add_request(p, sp) for p, sp in zip(prompts, plist)]
        while self.has_work():
            self.step()
        return [self.result(s) for s in seqs]

    def result(self, s: Sequence) -> dict:
        out = {
            "token_ids": s.completion_token_ids,
            "prompt_token_ids": s.token_ids[: s.num_prompt_tokens],
            "finish_reason": s.finish_reason,
            "num_preemptions": s.num_preemptions,
            "tau": s.tau,
            "spec_steps": s.spec_steps,
            "spec_accepted": s.spec_accepted,
            "logit_gaps": getattr(s, "logit_gaps", None),
            "ttft_s": (s.first_token_time - s.arrival_time) if s.first_token_time else None,
            "token_times": s.token_times,
            "arrival_time": s.arrival_time,
        }
        if self.tokenizer is not None:
            out["text"] = self.tokenizer.decode(s.completion_token_ids, skip_special_tokens=True)
        return out


class LLM(LLMEngine):
    """Convenience: LLM("Qwen/Qwen3-0.6B", max_num_seqs=64, ...)."""

    def __init__(self, model: str, **kwargs):
        super().__init__(EngineConfig(model=model, **kwargs))
