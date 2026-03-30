// UVM (Unified Virtual Memory) allocator for CUDAPluggableAllocator.
// Uses cudaMallocManaged/cudaFree to enable GPU memory oversubscription
// on systems with high-bandwidth CPU-GPU interconnects (NVLink, Grace Hopper).
//
// Enable with: VLLM_USE_UVM_ALLOCATOR=1
//
// On NVLink systems (IBM POWER9, Grace Hopper), managed memory allows
// transparent overflow to system RAM with hardware-coherent access.

#include <cuda_runtime_api.h>
#include <iostream>

extern "C" {

#define PY_SSIZE_T_CLEAN
#include <Python.h>

// CUDAPluggableAllocator interface: malloc(size, device, stream) -> ptr
void* uvm_malloc(ssize_t size, int device, cudaStream_t stream) {
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        std::cerr << "UVM allocator: cudaSetDevice failed: "
                  << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }

    void* ptr = nullptr;

    if (size <= 0) {
        err = cudaMalloc(&ptr, size);
        return (err == cudaSuccess) ? ptr : nullptr;
    }

    // Try cudaMallocManaged first. Falls back to cudaMalloc if it fails
    // (e.g. during CUDA graph capture where managed alloc is not allowed).
    err = cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
    if (err != cudaSuccess) {
        cudaGetLastError();  // clear the error
        err = cudaMalloc(&ptr, size);
        if (err != cudaSuccess) {
            std::cerr << "UVM allocator: both cudaMallocManaged and cudaMalloc "
                      << "failed for " << size << " bytes: "
                      << cudaGetErrorString(err) << std::endl;
            return nullptr;
        }
        return ptr;  // skip cudaMemAdvise for regular allocations
    }

    // Only set AccessedBy hint — tells the driver this GPU will access
    // the memory, enabling direct mapping via NVLink without page faults.
    // We intentionally do NOT set PreferredLocation=device, because that
    // forces all pages onto GPU initially and causes very slow eviction
    // (page migration) when VRAM fills up during oversubscription.
    err = cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, device);
    if (err != cudaSuccess) {
        std::cerr << "UVM allocator: cudaMemAdvise(AccessedBy) failed: "
                  << cudaGetErrorString(err) << std::endl;
    }

    return ptr;
}

// CUDAPluggableAllocator interface: free(ptr, size, device, stream)
void uvm_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        std::cerr << "UVM allocator: cudaFree failed: "
                  << cudaGetErrorString(err) << std::endl;
    }
}

// Python module boilerplate (needed for CUDAPluggableAllocator to dlopen)
static PyMethodDef module_methods[] = {
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef uvm_allocator_module = {
    PyModuleDef_HEAD_INIT, "uvm_allocator",
    "UVM-based allocator for GPU memory oversubscription", -1, module_methods
};

PyMODINIT_FUNC PyInit_uvm_allocator(void) {
    return PyModule_Create(&uvm_allocator_module);
}

}  // extern "C"
