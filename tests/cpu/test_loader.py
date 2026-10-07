"""Fused-projection weight loading (FIXPLAN D1)."""

import pytest
import torch
from safetensors.torch import load_file, save_file

from tinyserve.loader import load_weights
from tinyserve.models.qwen3 import Qwen3ForCausalLM


def _model(path):
    from transformers import AutoConfig

    return Qwen3ForCausalLM(AutoConfig.from_pretrained(path), dtype=torch.float32)


def test_fused_weights_are_concatenated_shards(tiny_model):
    path, hf = tiny_model
    m = _model(path)
    load_weights(m, path)
    for i, layer in enumerate(hf.model.layers):
        a = layer.self_attn
        ours = m.model.layers[i].self_attn.qkv_proj.weight
        assert torch.equal(ours, torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]).float())
        mlp = m.model.layers[i].mlp.gate_up_proj.weight
        assert torch.equal(mlp, torch.cat([layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight]).float())


def test_missing_shard_is_an_error(tiny_model, tmp_path):
    path, _ = tiny_model
    tensors = load_file(str(path / "model.safetensors"))
    del tensors["model.layers.1.self_attn.k_proj.weight"]
    save_file(tensors, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_bytes((path / "config.json").read_bytes())
    with pytest.raises(KeyError, match="qkv_proj"):
        load_weights(_model(tmp_path), tmp_path)
