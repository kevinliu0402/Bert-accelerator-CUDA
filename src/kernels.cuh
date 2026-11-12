// CUDA kernels for the BERT block. All matrices are row-major:
//   A [M][K] activations, W [K][N] weights (same layout as *_w_t.bin), C [M][N].
#pragma once

#include <cuda_runtime.h>

#include "fixed_point.h"

namespace bert {

enum Act : int { kActNone = 0, kActGelu = 1, kActTanh = 2 };

enum class GemmKind {
    Naive,   // one thread per output, all operands from global memory
    Tiled,   // 32x32 shared-memory tiles, one output per thread
    Reg32,   // 32x32 block tile, 2x2 outputs per thread (more blocks for small M)
    Reg64,   // 64x64 block tile, 4x4 outputs per thread (more reuse for large M)
    Cublas,  // cublasSgemm + fused bias/activation kernel (reference point)
};
constexpr int kNumGemmKinds = 5;

const char* gemm_name(GemmKind k);
bool parse_gemm(const char* s, GemmKind* k);

// C = act(A * W + bias)
void gemm_bias_act(GemmKind kind, const float* A, const float* W, const float* bias, float* C, int M, int N, int K,
                   int act, cudaStream_t s);

// Same, using the project's fixed-point arithmetic (bit-exact with the CPU model).
void gemm_bias_fixed(const float* A, const float* W, const float* bias, float* C, int M, int N, int K,
                     const FixedLayer& cfg, int act, cudaStream_t s);

// Masked multi-head attention for `batch` sentences of `seq` tokens.
// q/k/v/out: [batch*seq][heads*head_dim]; masks: device array [batch].
void attention(const float* q, const float* k, const float* v, float* out, int batch, int seq, int heads,
               int head_dim, const int* masks, cudaStream_t s);

void add(const float* a, const float* b, float* out, int n, cudaStream_t s);

// out[b][:] = in[b*row_stride][:] (pick the [CLS] row of each sentence)
void gather_rows(const float* in, float* out, int batch, int row_stride, int cols, cudaStream_t s);

// cuBLAS needs a workspace set before it can be captured into a CUDA graph.
void cublas_init(cudaStream_t s);
void cublas_shutdown();

}  // namespace bert
