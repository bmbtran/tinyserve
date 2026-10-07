"""One-off profile (PLAN.md R14): where does tinyserve lose to vLLM on W1?

1. Offline engine throughput at c=64 (no HTTP) vs the server number -> server overhead.
2. Per-step time split: schedule / model runner (tensor prep + GPU) / postprocess, by batch kind.
3. torch.profiler: top ops by self CPU time over a few decode-heavy steps.
"""

import time
from collections import defaultdict

import pytest
import torch

from bench.workloads import w1_random
from tests.util import closed_loop
from tinyserve.config import EngineConfig
from tinyserve.engine import LLMEngine
from tinyserve.sampling import SamplingParams

pytestmark = pytest.mark.gpu


def test_profile_w1(report):
    eng = LLMEngine(EngineConfig(model="Qwen/Qwen3-0.6B", dtype="bfloat16", device="cuda", attn_backend="flash",
                                 block_size=256, max_num_seqs=128, max_model_len=4096, max_num_batched_tokens=8192))
    reqs = w1_random()
    sp = SamplingParams(max_tokens=256, ignore_eos=True)
    closed_loop(eng, [r["prompt"] for r in reqs[:16]], 16, SamplingParams(max_tokens=16, ignore_eos=True))  # warmup

    # 1. offline closed loop at c=64 (same requests as the W1 benchmark)
    seqs, wall = closed_loop(eng, [r["prompt"] for r in reqs], 64, sp)
    tok_s = sum(s.num_completion_tokens for s in seqs) / wall
    report["offline_c64_output_tok_s"] = round(tok_s, 1)
    print(f"PROFILE offline engine c=64: {tok_s:.0f} output tok/s (server W1 c=64 is in results/bench)")

    # 2. step time split (synchronizing the GPU so 'runner' includes kernel time)
    split = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    orig_schedule, orig_run, orig_post = eng.scheduler.schedule, eng.runner.run, eng.scheduler.postprocess
    state = {}

    def timed(name, fn):
        def wrap(*a, **k):
            t = time.perf_counter()
            out = fn(*a, **k)
            torch.cuda.synchronize()
            state.setdefault("pending", []).append((name, time.perf_counter() - t, out))
            return out
        return wrap

    eng.scheduler.schedule = timed("schedule", orig_schedule)
    eng.runner.run = timed("runner", orig_run)
    eng.scheduler.postprocess = timed("postprocess", orig_post)
    queue = [r["prompt"] for r in reqs[:128]]
    live = 0
    t0 = time.perf_counter()
    while queue or eng.has_work():
        while queue and live < 64:
            eng.add_request(queue.pop(0), sp)
            live += 1
        batch, fin = eng.step()
        live -= len(fin)
        kind = batch.kind.name if batch else "NONE"
        counts[kind] += 1
        for name, dt_, _ in state.pop("pending", []):
            split[kind][name] += dt_
    total = time.perf_counter() - t0
    rows = {k: {n: round(v * 1000 / counts[k], 2) for n, v in d.items()} for k, d in split.items()}
    share = {k: round(sum(d.values()) / total, 3) for k, d in split.items()}
    report["step_ms_by_kind"] = rows
    report["steps_by_kind"] = dict(counts)
    report["time_share_by_kind"] = share
    for k in rows:
        print(f"PROFILE {k}: {counts[k]} steps, mean ms per step {rows[k]}, share of wall {share[k]:.1%}")
    eng.scheduler.schedule, eng.runner.run, eng.scheduler.postprocess = orig_schedule, orig_run, orig_post

    # 3. torch.profiler over ~40 steps of a decode-heavy window
    for p in [r["prompt"] for r in reqs[:64]]:
        eng.add_request(p, SamplingParams(max_tokens=60, ignore_eos=True))
    for _ in range(3):
        eng.step()  # prefills
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(40):
            eng.step()
        torch.cuda.synchronize()
    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15)
    print(table)
    report["profiler_top_cpu"] = table.splitlines()[:20]
    while eng.has_work():
        eng.step()
