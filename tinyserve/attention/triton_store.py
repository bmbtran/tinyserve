"""The one Triton kernel in tinyserve: scatter new K/V rows into cache slots.

Why a kernel at all? Storing K/V is "for each new token i, copy its K and V
(num_kv_heads * head_dim numbers each) into cache slot slot_mapping[i]".
PyTorch can do it with index_copy_, but slots of -1 ("do not store") would
need a boolean mask, which is a data-dependent shape. This kernel skips them
for free and is safe to capture in a CUDA graph.

Triton primer (enough to read the kernel below):
  * A kernel launch starts a grid of *programs* that run in parallel. Here
    the grid is (N,), one program per token; `tl.program_id(0)` tells each
    program which token it owns.
  * `tl.arange(0, D)` is a vector [0, 1, ..., D-1]. Adding it to a pointer
    gives D pointers, so one `tl.load` reads D consecutive numbers at once
    (D must be a power of two, a compile-time constant: `tl.constexpr`).
  * Pointer arithmetic is in elements, not bytes: row i of a 2-D tensor with
    row stride s starts at `ptr + i * s`.
  * A `mask=` argument on load/store disables lanes that would be out of
    bounds; we do not need one because every row has exactly D elements.
  * The early `return` when slot == -1 makes that program do nothing.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU machines
    triton = None

if triton is not None:

    @triton.jit
    def _store_kv_kernel(k_ptr, k_row_stride, v_ptr, v_row_stride, k_cache_ptr, v_cache_ptr, slot_ptr, D: tl.constexpr):
        i = tl.program_id(0)  # which token this program copies
        slot = tl.load(slot_ptr + i)
        if slot == -1:
            return
        offs = tl.arange(0, D)  # the D = num_kv_heads * head_dim values of one token
        k = tl.load(k_ptr + i * k_row_stride + offs)
        v = tl.load(v_ptr + i * v_row_stride + offs)
        tl.store(k_cache_ptr + slot * D + offs, k)
        tl.store(v_cache_ptr + slot * D + offs, v)


def triton_store_kv(k: torch.Tensor, v: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor) -> None:
    """k, v: [N, Hkv, Dh] (each row contiguous); caches: [blocks, block_size, Hkv, Dh]."""
    if triton is None:
        raise RuntimeError("triton is not available")
    n, hkv, dh = k.shape
    d = hkv * dh
    assert k.stride(-1) == 1 and k.stride(1) == dh and v.stride(-1) == 1 and v.stride(1) == dh
    assert k_cache.is_contiguous() and v_cache.is_contiguous() and slot_mapping.numel() == n
    assert d & (d - 1) == 0, "num_kv_heads * head_dim must be a power of two"
    _store_kv_kernel[(n,)](k, k.stride(0), v, v.stride(0), k_cache, v_cache, slot_mapping, D=d)
