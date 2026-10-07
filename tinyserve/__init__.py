"""tinyserve: a small LLM inference engine for Qwen3."""

from tinyserve.config import EngineConfig
from tinyserve.sampling import SamplingParams

__all__ = ["LLM", "LLMEngine", "EngineConfig", "SamplingParams"]


def __getattr__(name):
    # Lazy so `import tinyserve.block_manager` stays light (no model code).
    if name in ("LLM", "LLMEngine"):
        from tinyserve import engine

        return getattr(engine, name)
    raise AttributeError(name)
