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
    a single shared parameter, so they are filled once."""
    params = dict(module.named_parameters())  # dedups tied params (lm_head.weight absent if tied)
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
