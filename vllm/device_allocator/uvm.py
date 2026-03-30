# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UVM (Unified Virtual Memory) allocator for GPU memory oversubscription.

Uses cudaMallocManaged via PyTorch's CUDAPluggableAllocator to allow
transparent overflow to system RAM on systems with high-bandwidth
CPU-GPU interconnects (IBM POWER9 NVLink, Grace Hopper).

Only model weights are allocated with managed memory. Activations, KV cache,
and CUDA graph buffers use PyTorch's default caching allocator.

Enable with: VLLM_USE_UVM_ALLOCATOR=1
"""

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from vllm.logger import init_logger
from vllm.utils.system_utils import find_loaded_library

logger = init_logger(__name__)

_lib_path: str | None = None


def _get_lib_path() -> str:
    global _lib_path
    if _lib_path is not None:
        return _lib_path

    try:
        import vllm.uvm_allocator  # noqa: F401
    except ModuleNotFoundError:
        raise RuntimeError(
            "UVM allocator extension not found. "
            "Make sure vLLM was built with CUDA support."
        )

    _lib_path = find_loaded_library("uvm_allocator")
    if _lib_path is None:
        raise RuntimeError(
            "uvm_allocator shared library loaded but path not found"
        )
    return _lib_path


@contextmanager
def uvm_memory_pool() -> Iterator[None]:
    """Context manager that routes allocations through cudaMallocManaged.

    Only allocations made inside this context use managed memory.
    Everything outside (activations, KV cache, CUDA graphs) uses
    PyTorch's default caching allocator.
    """
    lib_path = _get_lib_path()
    alloc = torch.cuda.memory.CUDAPluggableAllocator(
        lib_path, "uvm_malloc", "uvm_free"
    )
    pool = torch.cuda.memory.MemPool(alloc._allocator)

    logger.info(
        "Using UVM allocator (cudaMallocManaged) for this allocation scope. "
        "Memory will transparently overflow to system RAM."
    )

    with torch.cuda.memory.use_mem_pool(pool):
        yield
