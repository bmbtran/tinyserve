import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_collection_modifyitems(config, items):
    import torch

    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="needs a CUDA GPU (runs on Modal)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    """(path, hf_model): a 2-layer random float64 Qwen3 saved as safetensors."""
    from tests.util import make_tiny_model

    path = tmp_path_factory.mktemp("tiny_qwen3")
    return path, make_tiny_model(path)
