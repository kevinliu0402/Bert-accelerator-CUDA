#include "kernels.cuh"

#include <cstring>

#include "cuda_check.h"

namespace bert {

const char* gemm_name(GemmKind k) {
    switch (k) {
        case GemmKind::Naive: return "naive";
        case GemmKind::Tiled: return "tiled";
        case GemmKind::Reg32: return "reg32";
        case GemmKind::Reg64: return "reg64";
        case GemmKind::Cublas: return "cublas";
    }
    return "?";
}

bool parse_gemm(const char* s, GemmKind* k) {
    for (int i = 0; i < kNumGemmKinds; ++i)
        if (std::strcmp(s, gemm_name((GemmKind)i)) == 0) {
            *k = (GemmKind)i;
            return true;
        }
    return false;
}

__device__ __forceinline__ float act_fn(float x, int act) {
    if (act == kActGelu) {
        const float kC = 0.7978845608028654f;  // sqrt(2/pi)
        return 0.5f * x * (1.0f + tanhf(kC * (x + 0.044715f * x * x * x)));
    }
    if (act == kActTanh) return tanhf(x);
    return x;
}

// ---------------------------------------------------------------------------
// 1. Naive: one thread per output. Threads in a warp share the row of A and
//    read consecutive columns of W, so W loads coalesce but nothing is reused.
// ---------------------------------------------------------------------------
__global__ void gemm_naive_kernel(const float* __restrict__ A, const float* __restrict__ W,
                                  const float* __restrict__ bias, float* __restrict__ C, int M, int N, int K,
                                  int act) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[(size_t)m * K + k] * W[(size_t)k * N + n];
    C[(size_t)m * N + n] = act_fn(acc + bias[n], act);
}

// ---------------------------------------------------------------------------
// 2. Shared-memory tiling: each 32x32 block stages a tile of A and W in shared
//    memory, so every global element is read once per tile instead of 32 times.
// ---------------------------------------------------------------------------
constexpr int kTile = 32;

__global__ void gemm_tiled_kernel(const float* __restrict__ A, const float* __restrict__ W,
                                  const float* __restrict__ bias, float* __restrict__ C, int M, int N, int K,
                                  int act) {
    __shared__ float As[kTile][kTile];
    __shared__ float Ws[kTile][kTile + 1];  // +1 pad: no bank conflicts on column reads
    int tx = threadIdx.x, ty = threadIdx.y;
    int m = blockIdx.y * kTile + ty;
    int n = blockIdx.x * kTile + tx;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += kTile) {
        As[ty][tx] = (m < M && k0 + tx < K) ? A[(size_t)m * K + k0 + tx] : 0.0f;
        Ws[ty][tx] = (k0 + ty < K && n < N) ? W[(size_t)(k0 + ty) * N + n] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < kTile; ++kk) acc += As[ty][kk] * Ws[kk][tx];
        __syncthreads();
    }
    if (m < M && n < N) C[(size_t)m * N + n] = act_fn(acc + bias[n], act);
}

// ---------------------------------------------------------------------------
// 3. Register blocking: each thread keeps a TM x TN block of outputs in
//    registers and reuses every shared-memory value TM or TN times.
//    BM x BN is the block tile, BK the K-slice staged per iteration.
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
gemm_regblock_kernel(const float* __restrict__ A, const float* __restrict__ W, const float* __restrict__ bias,
                     float* __restrict__ C, int M, int N, int K, int act) {
    constexpr int TX = BN / TN;  // threads along N
    constexpr int NT = (BM / TM) * TX;
    __shared__ float As[BK][BM + 1];  // stored K-major so a thread's TM rows are contiguous
    __shared__ float Ws[BK][BN];

    const int tid = threadIdx.x;
    const int tx = tid % TX, ty = tid / TX;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Coalesced loads: consecutive threads read consecutive addresses.
        for (int i = tid; i < BM * BK; i += NT) {
            int r = i / BK, c = i % BK;
            int gr = row0 + r, gc = k0 + c;
            As[c][r] = (gr < M && gc < K) ? A[(size_t)gr * K + gc] : 0.0f;
        }
        for (int i = tid; i < BK * BN; i += NT) {
            int r = i / BN, c = i % BN;
            int gr = k0 + r, gc = col0 + c;
            Ws[r][c] = (gr < K && gc < N) ? W[(size_t)gr * N + gc] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a[TM], w[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[kk][ty * TM + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) w[j] = Ws[kk][tx + j * TX];  // strided columns: conflict-free
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * w[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        int r = row0 + ty * TM + i;
        if (r >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            int c = col0 + tx + j * TX;  // consecutive threads write consecutive columns
            if (c < N) C[(size_t)r * N + c] = act_fn(acc[i][j] + bias[c], act);
        }
    }
}

__global__ void bias_act_kernel(float* C, const float* __restrict__ bias, int M, int N, int act) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M * N) C[i] = act_fn(C[i] + bias[i % N], act);
}

// ---------------------------------------------------------------------------
// Fixed point: one thread per output, accumulating in the same order and with
// the same per-step saturation as the CPU model, so results are bit-exact.
// ---------------------------------------------------------------------------
__global__ void gemm_fixed_kernel(const float* __restrict__ A, const float* __restrict__ W,
                                  const float* __restrict__ bias, float* __restrict__ C, int M, int N, int K,
                                  FixedLayer cfg, int act) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if (m >= M || n >= N) return;
    float v = fx_dot_bias(A + (size_t)m * K, W, N, n, bias[n], K, cfg);
    C[(size_t)m * N + n] = act_fn(v, act);
}

// ---------------------------------------------------------------------------
// Attention: one block per (head, query token, sentence). blockDim.x = head_dim.
// Uses the project's softmax variant exp(1 + score/8) with masked keys removed.
// ---------------------------------------------------------------------------
__global__ void attention_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                 const float* __restrict__ v, float* __restrict__ out, int seq, int heads,
                                 int head_dim, const int* __restrict__ masks) {
    extern __shared__ float sm[];
    float* qrow = sm;              // [head_dim]
    float* p = sm + head_dim;      // [seq]
    __shared__ float inv_sum;

    const int h = blockIdx.x, s = blockIdx.y, b = blockIdx.z;
    const int D = heads * head_dim;
    const int mask = masks[b];
    const size_t base = (size_t)b * seq * D;
    const int d = threadIdx.x;

    qrow[d] = q[base + (size_t)s * D + h * head_dim + d];
    __syncthreads();

    for (int t = threadIdx.x; t < seq; t += blockDim.x) {
        const float* krow = k + base + (size_t)t * D + h * head_dim;
        float score = 0.0f;
        for (int j = 0; j < head_dim; ++j) score += qrow[j] * krow[j];
        p[t] = t < mask ? expf(1.0f + score / 8.0f) : 0.0f;
    }
    __syncthreads();
    if (threadIdx.x == 0) {  // seq is 32: a serial sum is cheaper than a tree here
        float sum = 0.0f;
        for (int t = 0; t < mask && t < seq; ++t) sum += p[t];
        inv_sum = 1.0f / sum;
    }
    __syncthreads();

    float acc = 0.0f;
    for (int t = 0; t < seq; ++t) acc += p[t] * inv_sum * v[base + (size_t)t * D + h * head_dim + d];
    out[base + (size_t)s * D + h * head_dim + d] = acc;
}

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void gather_rows_kernel(const float* in, float* out, int row_stride, int cols) {
    int b = blockIdx.x;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) out[(size_t)b * cols + c] = in[(size_t)b * row_stride * cols + c];
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------
static cublasHandle_t g_cublas = nullptr;
static void* g_cublas_ws = nullptr;

void cublas_init(cudaStream_t s) {
    if (!g_cublas) {
        CUBLAS_CHECK(cublasCreate(&g_cublas));
        const size_t ws = 4 << 20;
        CUDA_CHECK(cudaMalloc(&g_cublas_ws, ws));
        CUBLAS_CHECK(cublasSetWorkspace(g_cublas, g_cublas_ws, ws));
        // Keep FP32 math exact (no TF32) so results compare against the CPU model.
        CUBLAS_CHECK(cublasSetMathMode(g_cublas, CUBLAS_DEFAULT_MATH));
    }
    CUBLAS_CHECK(cublasSetStream(g_cublas, s));
}

void cublas_shutdown() {
    if (g_cublas) cublasDestroy(g_cublas);
    if (g_cublas_ws) cudaFree(g_cublas_ws);
    g_cublas = nullptr;
    g_cublas_ws = nullptr;
}

template <int BM, int BN, int BK, int TM, int TN>
static void launch_regblock(const float* A, const float* W, const float* bias, float* C, int M, int N, int K, int act,
                            cudaStream_t s) {
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_regblock_kernel<BM, BN, BK, TM, TN><<<grid, (BM / TM) * (BN / TN), 0, s>>>(A, W, bias, C, M, N, K, act);
}

void gemm_bias_act(GemmKind kind, const float* A, const float* W, const float* bias, float* C, int M, int N, int K,
                   int act, cudaStream_t s) {
    switch (kind) {
        case GemmKind::Naive: {
            dim3 block(32, 8), grid((N + 31) / 32, (M + 7) / 8);
            gemm_naive_kernel<<<grid, block, 0, s>>>(A, W, bias, C, M, N, K, act);
            break;
        }
        case GemmKind::Tiled: {
            dim3 block(kTile, kTile), grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
            gemm_tiled_kernel<<<grid, block, 0, s>>>(A, W, bias, C, M, N, K, act);
            break;
        }
        case GemmKind::Reg32: launch_regblock<32, 32, 32, 2, 2>(A, W, bias, C, M, N, K, act, s); break;
        case GemmKind::Reg64: launch_regblock<64, 64, 16, 4, 4>(A, W, bias, C, M, N, K, act, s); break;
        case GemmKind::Cublas: {
            cublas_init(s);
            const float one = 1.0f, zero = 0.0f;
            // Row-major C = A*W is column-major C^T = W^T * A^T: swap operands, no transposes.
            CUBLAS_CHECK(cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, W, N, A, K, &zero, C, N));
            int n = M * N;
            bias_act_kernel<<<(n + 255) / 256, 256, 0, s>>>(C, bias, M, N, act);
            break;
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

void gemm_bias_fixed(const float* A, const float* W, const float* bias, float* C, int M, int N, int K,
                     const FixedLayer& cfg, int act, cudaStream_t s) {
    dim3 grid((N + 127) / 128, M);
    gemm_fixed_kernel<<<grid, 128, 0, s>>>(A, W, bias, C, M, N, K, cfg, act);
    CUDA_CHECK(cudaGetLastError());
}

void attention(const float* q, const float* k, const float* v, float* out, int batch, int seq, int heads,
               int head_dim, const int* masks, cudaStream_t s) {
    dim3 grid(heads, seq, batch);
    size_t shm = sizeof(float) * (head_dim + seq);
    attention_kernel<<<grid, head_dim, shm, s>>>(q, k, v, out, seq, heads, head_dim, masks);
    CUDA_CHECK(cudaGetLastError());
}

void add(const float* a, const float* b, float* out, int n, cudaStream_t s) {
    add_kernel<<<(n + 255) / 256, 256, 0, s>>>(a, b, out, n);
    CUDA_CHECK(cudaGetLastError());
}

void gather_rows(const float* in, float* out, int batch, int row_stride, int cols, cudaStream_t s) {
    gather_rows_kernel<<<batch, 128, 0, s>>>(in, out, row_stride, cols);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace bert
