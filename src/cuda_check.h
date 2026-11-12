#pragma once

#include <cstdio>
#include <cstdlib>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                              \
    do {                                                                                              \
        cudaError_t err_ = (call);                                                                    \
        if (err_ != cudaSuccess) {                                                                    \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_), __FILE__,    \
                         __LINE__, cudaGetErrorString(err_));                                         \
            std::exit(1);                                                                             \
        }                                                                                             \
    } while (0)

#define CUBLAS_CHECK(call)                                                                       \
    do {                                                                                         \
        cublasStatus_t st_ = (call);                                                             \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                                      \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__, __LINE__);    \
            std::exit(1);                                                                        \
        }                                                                                        \
    } while (0)

// Times work enqueued on a stream with CUDA events.
struct GpuTimer {
    cudaEvent_t a, b;
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&a));
        CUDA_CHECK(cudaEventCreate(&b));
    }
    ~GpuTimer() {
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }
    void start(cudaStream_t s) { CUDA_CHECK(cudaEventRecord(a, s)); }
    float stop_ms(cudaStream_t s) {
        CUDA_CHECK(cudaEventRecord(b, s));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        return ms;
    }
};
