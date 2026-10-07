"""Engine configuration.

One dataclass holds every knob. `validate()` catches bad combinations early
(for example a flash-attn block size that is not a multiple of 256).
"""

from dataclasses import dataclass


@dataclass
class EngineConfig:
    model: str = "Qwen/Qwen3-0.6B"
    dtype: str = "bfloat16"  # "float32" / "float64" allowed for CPU tests
    device: str = "cuda"  # "cpu" for local tests
    attn_backend: str = "flash"  # "flash" | "torch"
    block_size: int = 256  # flash backend needs % 256 == 0; torch backend any >= 1
    num_kv_blocks: int | None = None  # None -> derive from gpu_memory_utilization
    gpu_memory_utilization: float = 0.85
    max_num_seqs: int = 128
    max_num_batched_tokens: int = 8192
    max_model_len: int = 4096
    enable_prefix_cache: bool = True
    enforce_eager: bool = False  # True disables CUDA graphs
    cuda_graph_batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    spec_method: str | None = None  # None | "dflash" | "oracle" | "random" (last two: tests only)
    spec_draft_model: str | None = None  # e.g. "z-lab/Qwen3-4B-DFlash-b16"
    spec_block_size: int = 16  # DFlash block = 1 real token + 15 mask tokens
    use_triton_store: bool = True  # False -> torch index_copy_ store (always False on CPU)
    seed: int = 0

    @property
    def spec_lookahead(self) -> int:
        """Extra KV slots a SPEC step needs past the last token (0 without spec)."""
        return self.spec_block_size - 1 if self.spec_method else 0

    @property
    def max_blocks_per_seq(self) -> int:
        """Width of a block table: room for max_model_len tokens plus the spec lookahead."""
        return (self.max_model_len + self.spec_block_size + self.block_size - 1) // self.block_size

    def validate(self) -> "EngineConfig":
        if self.attn_backend not in ("flash", "torch"):
            raise ValueError(f"unknown attn_backend {self.attn_backend!r}")
        if self.attn_backend == "flash" and self.block_size % 256 != 0:
            raise ValueError("flash-attn paged KV needs block_size % 256 == 0")
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if self.spec_method not in (None, "dflash", "oracle", "random"):
            raise ValueError(f"unknown spec_method {self.spec_method!r}")
        if self.spec_method == "dflash" and not self.spec_draft_model:
            raise ValueError("spec_method='dflash' needs spec_draft_model")
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError("max_num_batched_tokens must be >= max_model_len (no chunked prefill)")
        if self.device == "cpu":
            # Triton and CUDA graphs only exist on the GPU.
            self.use_triton_store = False
            self.enforce_eager = True
        return self
