"""Load safetensors checkpoints into tinyserve modules by parameter name."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open


def resolve_model_path(name_or_path: str) -> Path:
    """A local directory, or a Hugging Face repo id resolved through the HF
    cache (downloading if allowed; HF_HUB_OFFLINE=1 makes it cache-only)."""
    p = Path(name_or_path)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        name_or_path,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.py"],
        local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
    ))


@torch.no_grad()
def load_weights(module: nn.Module, path: Path, prefix_map: dict[str, str] | None = None, strict: bool = True) -> None:
    """Copy every tensor in path/*.safetensors into the parameter of the same
    name (after optional prefix renames). Tied weights (lm_head = embed) are
    a single shared parameter, so they are filled once.

    Fused projections: a submodule with `shard_sizes = {fused: {ckpt_name: (row_start, rows)}}`
    receives e.g. `...self_attn.q_proj.weight` into rows of `...self_attn.qkv_proj.weight`."""
    params = dict(module.named_parameters())  # dedups tied params (lm_head.weight absent if tied)
    routes: dict[str, tuple[str, int, int]] = {}  # checkpoint name -> (fused param, row start, rows)
    shards_needed: dict[str, int] = {}
    for prefix, sub in module.named_modules():
        for fused, shards in getattr(sub, "shard_sizes", {}).items():
            for suffix in ("weight", "bias"):
                target = f"{prefix}.{fused}.{suffix}" if prefix else f"{fused}.{suffix}"
                if target not in params:
                    continue
                shards_needed[target] = len(shards)
                for ckpt, (start, rows) in shards.items():
                    routes[f"{prefix}.{ckpt}.{suffix}" if prefix else f"{ckpt}.{suffix}"] = (target, start, rows)
    shards_seen: dict[str, int] = {}
    loaded: set[str] = set()
    files = sorted(Path(path).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {path}")
    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as st:
            for key in st.keys():
                name = key
                for old, new in (prefix_map or {}).items():
                    if name.startswith(old):
                        name = new + name[len(old):]
                if name in routes:
                    target, start, rows = routes[name]
                    t = st.get_tensor(key)
                    dst = params[target].narrow(0, start, rows)
                    if dst.shape != t.shape:
                        raise ValueError(f"shape mismatch for {name} -> {target}: {tuple(t.shape)} vs {tuple(dst.shape)}")
                    dst.copy_(t.to(dst.dtype))
                    shards_seen[target] = shards_seen.get(target, 0) + 1
                    if shards_seen[target] == shards_needed[target]:
                        loaded.add(target)
                    continue
                if name not in params:
                    if name == "lm_head.weight":  # tied checkpoints sometimes store it anyway
                        continue
                    if strict:
                        raise KeyError(f"unexpected weight {key} in {f.name}")
                    continue
                t = st.get_tensor(key)
                p = params[name]
                if p.shape != t.shape:
                    raise ValueError(f"shape mismatch for {name}: {tuple(t.shape)} vs {tuple(p.shape)}")
                p.copy_(t.to(p.dtype))
                loaded.add(name)
    missing = set(params) - loaded
    if missing and strict:
        raise KeyError(f"weights missing from checkpoint: {sorted(missing)[:10]}")
