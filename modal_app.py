"""All Modal functions and local entrypoints for tinyserve (single file).

Cost rules (PLAN.md 8.3): only `modal run` (ephemeral), explicit timeouts,
scaledown_window=30, max_containers=1, a single GPU type (L4), weights only
from the `tinyserve-hf-cache` volume, and `scripts/cost_guard.py` before every
GPU entrypoint. Every GPU entrypoint appends a COSTLOG.md row itself.

    uv run modal run modal_app.py::download_models
    uv run modal run modal_app.py::smoke
    uv run modal run modal_app.py::gpu_tests --suite m2
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
GPU = "L4"
HF_MOUNT = "/hf"
DEFAULT_REPOS = "Qwen/Qwen3-0.6B,Qwen/Qwen3-4B,z-lab/Qwen3-4B-DFlash-b16"

FA_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/"
    "flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)

app = modal.App("tinyserve")
volume = modal.Volume.from_name("tinyserve-hf-cache", create_if_missing=True)

HF_ENV = {"HF_HUB_CACHE": f"{HF_MOUNT}/hub", "HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false"}

# Small CPU image just for downloading weights.
download_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("huggingface_hub[hf_xet]").env(HF_ENV)

# Engine image: pip layers first (built once), local code mounted last (no rebuild on edits).
engine_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.8.0",
        "transformers==4.57.3",
        "safetensors",
        "xxhash",
        "numpy",
        "fastapi",
        "uvicorn",
        "httpx",
        "pytest",
        "huggingface_hub[hf_xet]",
        "openai",
    )
    .uv_pip_install(FA_WHEEL)
    .env({**HF_ENV, "HF_HUB_OFFLINE": "1"})
    .add_local_file(ROOT / "pyproject.toml", "/root/pyproject.toml")
    .add_local_dir(ROOT / "tests", "/root/tests", ignore=["**/__pycache__"])
    .add_local_dir(ROOT / "bench", "/root/bench", ignore=["**/__pycache__"])
    .add_local_python_source("tinyserve")
)

GPU_KW = dict(
    gpu=GPU, cpu=4, memory=16384, volumes={HF_MOUNT: volume}, scaledown_window=30, max_containers=1,
)


# ---------------------------------------------------------------------------
# Local helpers (run on the laptop)
# ---------------------------------------------------------------------------
def _git_sha() -> str:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "tinyserve", "tests", "bench", "modal_app.py"],
                           cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return (sha or "nocommit") + ("-dirty" if dirty else "")


def _cost_guard() -> None:
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "cost_guard.py")], cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"cost_guard refused (exit {proc.returncode}); not starting GPU work")


def _costlog(command: str, wall_s: float, purpose: str) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import cost_guard  # noqa: PLC0415

    billed = cost_guard.billing_month_to_date()
    est = wall_s / 3600 * 1.12
    path = ROOT / "COSTLOG.md"
    if not path.exists():
        path.write_text(
            "# COSTLOG\n\nOne row per GPU run (written automatically by `modal_app.py`). "
            "est $ = wall minutes x $1.12/60 (L4 + 4 CPU + 16 GiB).\n\n"
            "| date (UTC) | command | GPU | wall min | est $ (wall×$1.12/60) | billing month-to-date $ | purpose / outcome |\n"
            "|---|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
    billed_s = f"{billed:.2f}" if billed is not None else "n/a"
    with path.open("a", encoding="utf-8") as f:
        f.write(f"| {now} | `{command}` | {GPU} | {wall_s / 60:.1f} | {est:.3f} | {billed_s} | {purpose} |\n")


# ---------------------------------------------------------------------------
# M0: downloads + smoke
# ---------------------------------------------------------------------------
@app.function(image=download_image, cpu=2, memory=4096, timeout=1800, volumes={HF_MOUNT: volume},
              scaledown_window=30, max_containers=1)
def download_models(repos: str = DEFAULT_REPOS):
    """CPU-only: snapshot_download each repo into the volume."""
    from huggingface_hub import snapshot_download

    for repo in [r.strip() for r in repos.split(",") if r.strip()]:
        t = time.time()
        path = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt", "*.model", "*.jinja"])
        size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
        print(f"{repo}: {size / 1e9:.2f} GB in {time.time() - t:.0f}s -> {path}")
    volume.commit()


@app.function(image=engine_image, timeout=600, **GPU_KW)
def smoke_gpu() -> str:
    import torch
    import triton
    import triton.language as tl

    lines = []
    log = lambda s: (print(s), lines.append(s))  # noqa: E731
    log(f"torch={torch.__version__} cuda={torch.version.cuda} triton={triton.__version__}")
    log(f"torch._C._GLIBCXX_USE_CXX11_ABI={torch._C._GLIBCXX_USE_CXX11_ABI}")
    log(f"GPU {torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
    log(subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"],
                       capture_output=True, text=True).stdout.strip())
    import flash_attn
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

    log(f"flash_attn={flash_attn.__version__} wheel={FA_WHEEL.rsplit('/', 1)[-1]}")

    torch.manual_seed(0)
    dev, dt_ = "cuda", torch.bfloat16
    nb, bs, hq, hkv, d = 8, 256, 16, 8, 128
    k_cache = torch.randn(nb, bs, hkv, d, device=dev, dtype=dt_)
    v_cache = torch.randn(nb, bs, hkv, d, device=dev, dtype=dt_)
    block_table = torch.tensor([[3, 5], [1, 7]], device=dev, dtype=torch.int32)
    lens = [300, 400]
    scale = d**-0.5

    def ref_attn(q, b, k_len, causal):  # q: [Lq, hq, d]
        pos = torch.arange(k_len, device=dev)
        slots = block_table[b][pos // bs].long() * bs + pos % bs
        k = k_cache.view(-1, hkv, d)[slots].float().repeat_interleave(hq // hkv, 1)
        v = v_cache.view(-1, hkv, d)[slots].float().repeat_interleave(hq // hkv, 1)
        s = torch.einsum("qhd,khd->hqk", q.float(), k) * scale
        if causal:
            qpos = torch.arange(k_len - q.shape[0], k_len, device=dev)
            s = s.masked_fill(pos[None, None, :] > qpos[None, :, None], float("-inf"))
        return torch.einsum("hqk,khd->qhd", s.softmax(-1), v)

    # 1) decode: one query per sequence against the paged cache
    q = torch.randn(2, 1, hq, d, device=dev, dtype=dt_)
    out = flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=torch.tensor(lens, device=dev, dtype=torch.int32),
                                  block_table=block_table, softmax_scale=scale, causal=True)
    err = max((out[b, 0].float() - ref_attn(q[b], b, lens[b], True)).abs().max().item() for b in range(2))
    log(f"FA_OK paged_kvcache max_abs_err={err:.2e} {'<1e-2' if err < 1e-2 else '>=1e-2 FAIL'} vs torch reference")

    # 2) varlen prefill-with-prefix through block_table (causal) and non-causal
    qlens = [5, 7]
    qv = torch.randn(sum(qlens), hq, d, device=dev, dtype=dt_)
    cu_q = torch.tensor([0, 5, 12], device=dev, dtype=torch.int32)
    cu_k = torch.tensor([0, 300, 700], device=dev, dtype=torch.int32)
    ok = True
    for causal in (True, False):
        o = flash_attn_varlen_func(qv, k_cache, v_cache, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=7,
                                   max_seqlen_k=400, softmax_scale=scale, causal=causal, block_table=block_table)
        e = max((o[cu_q[b]:cu_q[b + 1]].float() - ref_attn(qv[cu_q[b]:cu_q[b + 1]], b, lens[b], causal)).abs().max().item()
                for b in range(2))
        log(f"  varlen causal={causal} max_abs_err={e:.2e}")
        ok &= e < 1e-2
    log("FA_VARLEN_BLOCKTABLE_OK" if ok else "FA_VARLEN_BLOCKTABLE_FAIL")

    # 3) [VERIFY M5] CUDA-graph padding rows: cache_seqlens=1 pointing at block 0
    bt_pad = torch.tensor([[3, 5], [0, 0], [0, 0]], device=dev, dtype=torch.int32)
    qp = torch.randn(3, 1, hq, d, device=dev, dtype=dt_)
    op = flash_attn_with_kvcache(qp, k_cache, v_cache, cache_seqlens=torch.tensor([300, 1, 1], device=dev, dtype=torch.int32),
                                 block_table=bt_pad, softmax_scale=scale, causal=True)
    log(f"FA_PAD_ROWS finite={bool(torch.isfinite(op).all())} (cache_seqlens=1, block 0)")
    try:
        op0 = flash_attn_with_kvcache(qp, k_cache, v_cache, cache_seqlens=torch.tensor([300, 0, 0], device=dev, dtype=torch.int32),
                                      block_table=bt_pad, softmax_scale=scale, causal=True)
        log(f"FA_PAD_ROWS_ZERO_LEN ok finite_real_row={bool(torch.isfinite(op0[0]).all())}")
    except Exception as e:  # noqa: BLE001
        log(f"FA_PAD_ROWS_ZERO_LEN error: {e}")

    # 4) Triton JIT works
    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)

    x = torch.randn(1000, device=dev)
    y = torch.randn(1000, device=dev)
    z = torch.empty_like(x)
    add_kernel[(triton.cdiv(1000, 256),)](x, y, z, 1000, BLOCK=256)
    log("TRITON_OK" if torch.allclose(z, x + y) else "TRITON_FAIL")

    # weights present in the volume?
    from huggingface_hub import snapshot_download

    for repo in DEFAULT_REPOS.split(","):
        try:
            snapshot_download(repo, local_files_only=True)
            log(f"weights cached: {repo}")
        except Exception as e:  # noqa: BLE001
            log(f"weights MISSING: {repo} ({type(e).__name__})")
    return "\n".join(lines)


@app.local_entrypoint()
def smoke():
    _cost_guard()
    t = time.time()
    try:
        out = smoke_gpu.remote()
    finally:
        _costlog("modal run modal_app.py::smoke", time.time() - t, "M0 env smoke")
    path = ROOT / "results" / "verify" / "m0_smoke.log"
    path.write_text(out + "\n", encoding="utf-8")
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# GPU test suites: one pytest invocation per milestone in one container
# ---------------------------------------------------------------------------
@app.function(image=engine_image, timeout=1800, **GPU_KW)
def gpu_tests_remote(suite: str, git_sha: str, pytest_args: str = "") -> dict:
    os.chdir("/root")
    out_json = f"/tmp/{suite}.json"
    env = {**os.environ, "TS_RESULT_JSON": out_json, "TS_GIT_SHA": git_sha, "PYTHONUNBUFFERED": "1"}
    cmd = [sys.executable, "-m", "pytest", f"tests/gpu/test_{suite}_gpu.py", "-s", "-q", "-rA", *pytest_args.split()]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    # Watchdog: stop pytest before the function's hard timeout so the partial
    # result JSON (written after every test) still comes back.
    import threading

    killed = []
    timer = threading.Timer(1800 - 150, lambda: (killed.append(True), proc.terminate()))
    timer.start()
    log_lines = []
    for line in proc.stdout:
        print(line, end="")
        log_lines.append(line)
    rc = proc.wait()
    timer.cancel()
    if killed:
        log_lines.append("\nWATCHDOG: pytest terminated before the 1800 s function timeout\n")
    result = json.loads(Path(out_json).read_text()) if Path(out_json).exists() else None
    return {"returncode": rc, "log": "".join(log_lines), "result": result}


@app.local_entrypoint()
def gpu_tests(suite: str, pytest_args: str = ""):
    _cost_guard()
    sha = _git_sha()
    t = time.time()
    status = "ERROR"
    try:
        res = gpu_tests_remote.remote(suite, sha, pytest_args)
        result = res["result"] or {"status": "FAIL", "error": "no result json written"}
        result.setdefault("status", "PASS" if res["returncode"] == 0 else "FAIL")
        result["git_sha"] = sha
        result["pytest_returncode"] = res["returncode"]
        status = result["status"]
        vdir = ROOT / "results" / "verify"
        (vdir / f"{suite}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        (vdir / f"{suite}.log").write_text(res["log"], encoding="utf-8")
        print(f"\nwrote results/verify/{suite}.json and .log; status={status}")
    finally:
        _costlog(f"modal run modal_app.py::gpu_tests --suite {suite}", time.time() - t, f"{suite.upper()} GPU tests: {status}")


# ---------------------------------------------------------------------------
# M8: benchmarks (server + bench/client.py in ONE container per engine)
# ---------------------------------------------------------------------------
VLLM_VERSION = "0.30.0"
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "httpx")
    .env({**HF_ENV, "HF_HUB_OFFLINE": "1"})
    .add_local_dir(ROOT / "bench", "/root/bench", ignore=["**/__pycache__"])
)

M06, M4B = "Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B"
DFLASH = "z-lab/Qwen3-4B-DFlash-b16"
PORT = 8000
W1_CONC = [1, 4, 16, 64, 128]
W3_SWEEP = [1, 2, 4, 8, 16]


def _server_cmd(engine: str, model: str, prefix_cache: bool = True, dflash: bool = False) -> list[str]:
    if engine == "tinyserve":
        cmd = [sys.executable, "-m", "tinyserve.server.api", "--model", model, "--port", str(PORT),
               "--max-model-len", "4096", "--gpu-memory-utilization", "0.85", "--max-num-seqs", "128"]
        if not prefix_cache:
            cmd.append("--no-prefix-cache")
        if dflash:
            cmd += ["--spec-method", "dflash", "--spec-draft-model", DFLASH]
        return cmd
    cmd = ["vllm", "serve", model, "--port", str(PORT), "--max-model-len", "4096", "--gpu-memory-utilization", "0.85",
           "--max-num-seqs", "128", "--dtype", "bfloat16", "--served-model-name", model]
    if not prefix_cache:
        cmd.append("--no-enable-prefix-caching")
    if dflash:
        cmd += ["--speculative-config", json.dumps({"method": "dflash", "model": DFLASH, "num_speculative_tokens": 15})]
    return cmd


def _spec_counters(engine: str):
    """(accepted draft tokens, verify steps) so far, from the server's /metrics."""
    import httpx

    try:
        r = httpx.get(f"http://127.0.0.1:{PORT}/metrics", timeout=5)
    except httpx.HTTPError:
        return None
    if engine == "tinyserve":
        m = r.json()
        return float(m.get("spec_accepted") or 0), float(m.get("spec_steps") or 0)
    acc = drafts = None
    for line in r.text.splitlines():
        if line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            acc = (acc or 0) + float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_drafts_total"):
            drafts = (drafts or 0) + float(line.rsplit(" ", 1)[1])
    return None if acc is None or drafts is None else (acc, drafts)


def _run_points(engine, model, endpoint, reqs_for, concs, dflash=False, repeat=()):
    """One benchmark point per concurrency (+ repeats). reqs_for(c) -> request list."""
    import asyncio

    from bench.client import run_load, summarize

    points = []
    for conc, is_rep in [(c, False) for c in concs] + [(c, True) for c in repeat]:
        before = _spec_counters(engine) if dflash else None
        s = summarize(asyncio.run(run_load(f"http://127.0.0.1:{PORT}", reqs_for(conc), conc, model, endpoint)))
        after = _spec_counters(engine) if dflash else None
        tau = None
        if before and after and after[1] > before[1]:
            tau = 1 + (after[0] - before[0]) / (after[1] - before[1])
        s["tau"] = round(tau, 3) if tau else None
        s["repeat"] = is_rep
        print(f"  [{engine}{' dflash' if dflash else ''}] {endpoint} c={conc}{' (repeat)' if is_rep else ''}: "
              f"{s['output_tok_s']} tok/s TTFT p50 {s['ttft_ms']['p50']} ms TPOT p50 {s['tpot_ms']['p50']} ms "
              f"errors {s['errors']} tau {s['tau']}", flush=True)
        points.append(s)
    return points


def _with_server(cmd, fn):
    """Start a server subprocess, wait for /health, run fn(), stop it."""
    from bench.client import wait_for_health

    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    try:
        waited = wait_for_health(f"http://127.0.0.1:{PORT}", 1200, proc)
        print(f"  healthy after {waited:.0f}s", flush=True)
        return fn(), waited
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(5)  # let the GPU memory go


def _bench_suite(engine: str, suite: str, version: str) -> dict:
    import asyncio

    from bench.client import run_load
    from bench.workloads import SPEC_DATASETS, load_chat, w1_random, w2_shared_prefix

    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().split(",")
    out = {}

    def record(key, model, workload, config, points, startup_s):
        out[key] = {"engine": engine, "engine_version": version, "model": model, "gpu": gpu[0].strip(),
                    "driver": gpu[1].strip() if len(gpu) > 1 else None, "workload": workload, "config": config,
                    "points": points, "server_startup_s": round(startup_s, 1), "key": key,
                    "started_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}

    def warmup(model, endpoint, reqs):
        asyncio.run(run_load(f"http://127.0.0.1:{PORT}", reqs[:8], 8, model, endpoint))

    if suite == "core":
        w1 = [{**r, "ignore_eos": True} for r in w1_random()]
        w2 = [{**r, "ignore_eos": True} for r in w2_shared_prefix()]
        warm = [{**r, "ignore_eos": True, "max_tokens": 16} for r in w1_random(n=8, seed=99)]

        def core_on():
            warmup(M06, "completions", warm)
            p1 = _run_points(engine, M06, "completions", lambda c: w1, W1_CONC, repeat=(16,))
            p2 = _run_points(engine, M06, "completions", lambda c: w2, [16])
            return p1, p2

        (p1, p2), st = _with_server(_server_cmd(engine, M06), core_on)
        record("W1", M06, "W1", {"prefix_cache": True, "num_requests": 256, "input_len": "U[256,768]", "output_len": 256}, p1, st)
        w2cfg = {"groups": 8, "per_group": 16, "prefix": 2048, "suffix": "U[64,128]", "output_len": 128}
        record("W2_cacheon", M06, "W2", {"prefix_cache": True, **w2cfg}, p2, st)
        if engine == "tinyserve":  # PLAN.md 7: cache OFF only for tinyserve
            def core_off():
                warmup(M06, "completions", warm)
                return _run_points(engine, M06, "completions", lambda c: w2, [16])

            p3, st = _with_server(_server_cmd(engine, M06, prefix_cache=False), core_off)
            record("W2_cacheoff", M06, "W2", {"prefix_cache": False, **w2cfg}, p3, st)
    elif suite == "spec":
        data = {d: [{"messages": r["messages"], "max_tokens": 512} for r in load_chat(d)] for d in SPEC_DATASETS}

        def n_for(c):  # budget: fewer requests at low concurrency
            return min(32, 8 * c)

        for dflash in (False, True):
            def spec_run():
                warmup(M4B, "chat", [{**r, "max_tokens": 32} for r in data["gsm8k"]])
                pts = {"gsm8k": _run_points(engine, M4B, "chat", lambda c: data["gsm8k"][: n_for(c)], W3_SWEEP, dflash, repeat=(1,))}
                for d in ("humaneval", "mtbench"):
                    pts[d] = _run_points(engine, M4B, "chat", lambda c, d=d: data[d][: n_for(c)], [1, 16], dflash)
                return pts

            pts, st = _with_server(_server_cmd(engine, M4B, dflash=dflash), spec_run)
            tag = "dflash" if dflash else "base"
            for d, p in pts.items():
                record(f"W3-{d}_{tag}", M4B, f"W3-{d}", {"spec": "dflash" if dflash else None, "max_tokens": 512,
                                                      "num_requests": "min(32, 8*c)", "temperature": 0, "enable_thinking": False}, p, st)
    else:
        raise ValueError(suite)
    return out


@app.function(image=engine_image, timeout=2400, **GPU_KW)
def bench_tinyserve(suite: str, git_sha: str) -> dict:
    os.chdir("/root")
    return _bench_suite("tinyserve", suite, git_sha)


@app.function(image=vllm_image, timeout=2400, **GPU_KW)
def bench_vllm(suite: str) -> dict:
    os.chdir("/root")
    sys.path.insert(0, "/root")
    import vllm

    return _bench_suite("vllm", suite, vllm.__version__)


@app.function(image=vllm_image, timeout=900, **GPU_KW)
def vllm_smoke_remote(dflash: bool = False) -> str:
    """M8a: start `vllm serve`, send one request (<= ~5 min)."""
    import httpx

    os.chdir("/root")
    sys.path.insert(0, "/root")
    import vllm

    model = M4B if dflash else M06

    def one():
        r = httpx.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", timeout=300, json={
            "model": model, "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}], "max_tokens": 32,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
        return r.status_code, r.json()["choices"][0]["message"]["content"], _spec_counters("vllm")

    (code, text, spec), waited = _with_server(_server_cmd("vllm", model, dflash=dflash), one)
    msg = f"vllm {vllm.__version__} model={model} dflash={dflash} startup={waited:.0f}s status={code} text={text!r} spec_counters={spec}"
    print(msg)
    return msg


@app.local_entrypoint()
def bench(engine: str, suite: str):
    _cost_guard()
    t = time.time()
    status = "ERROR"
    try:
        out = bench_tinyserve.remote(suite, _git_sha()) if engine == "tinyserve" else bench_vllm.remote(suite)
        for key, res in out.items():
            gpu = res["gpu"].replace("NVIDIA ", "").replace(" ", "")
            path = ROOT / "results" / "bench" / f"{engine}_{res['model'].split('/')[-1]}_{key}_{gpu}_{res['started_utc'][:10].replace('-', '')}.json"
            path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)}")
        status = "ok"
    finally:
        _costlog(f"modal run modal_app.py::bench --engine {engine} --suite {suite}", time.time() - t, f"M8 bench {engine}/{suite}: {status}")


@app.local_entrypoint()
def vllm_smoke(dflash: bool = False):
    _cost_guard()
    t = time.time()
    status = "ERROR"
    try:
        msg = vllm_smoke_remote.remote(dflash)
        (ROOT / "results" / "bench" / f"vllm_smoke{'_dflash' if dflash else ''}.log").write_text(msg + "\n", encoding="utf-8")
        status = "ok"
    finally:
        _costlog(f"modal run modal_app.py::vllm_smoke{' --dflash' if dflash else ''}", time.time() - t, f"M8a vLLM smoke: {status}")
