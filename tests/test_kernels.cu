// Unit tests for the CUDA kernels against the CPU reference.
//   * every FP32 GEMM implementation, with and without GELU, on shapes that are
//     and are not multiples of the tile sizes
//   * the fixed-point GEMM, which must be bit-exact with the CPU model
//   * masked attention
// Exits non-zero if anything fails.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "cpu_ref.h"
#include "cuda_check.h"
#include "kernels.cuh"

using namespace bert;

static int g_failures = 0, g_checks = 0;

static float* to_device(const std::vector<float>& h) {
    float* d;
    CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

static std::vector<float> to_host(const float* d, size_t n) {
    std::vector<float> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

static std::vector<float> random_vec(size_t n, float lo, float hi, std::mt19937& rng) {
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = u(rng);
    return v;
}

static void report(bool ok, const char* what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("  FAIL %s\n", what);
    }
}

static void test_gemm_fp32(std::mt19937& rng) {
    struct Shape { int M, N, K; };
    const Shape shapes[] = {{32, 512, 128}, {32, 128, 512}, {320, 512, 128}, {1, 128, 128}, {1, 2, 128},
                            {33, 130, 67},  {7, 5, 3},      {100, 77, 513},  {64, 64, 64}};
    cudaStream_t s = 0;
    cublas_init(s);
    for (const Shape& sh : shapes) {
        auto A = random_vec((size_t)sh.M * sh.K, -1, 1, rng);
        auto W = random_vec((size_t)sh.K * sh.N, -0.2f, 0.2f, rng);
        auto b = random_vec(sh.N, -0.5f, 0.5f, rng);
        float *dA = to_device(A), *dW = to_device(W), *db = to_device(b), *dC;
        CUDA_CHECK(cudaMalloc(&dC, (size_t)sh.M * sh.N * sizeof(float)));
        for (int act : {kActNone, kActGelu}) {
            std::vector<float> ref((size_t)sh.M * sh.N);
            gemm_bias_cpu(A.data(), W.data(), b.data(), ref.data(), sh.M, sh.N, sh.K, act);
            // Error bound scales with sum_k |a*w| (FP32 rounding in a different order).
            std::vector<float> mag((size_t)sh.M * sh.N, 0.0f);
            for (int m = 0; m < sh.M; ++m)
                for (int n = 0; n < sh.N; ++n) {
                    float t = std::fabs(b[n]);
                    for (int k = 0; k < sh.K; ++k) t += std::fabs(A[(size_t)m * sh.K + k] * W[(size_t)k * sh.N + n]);
                    mag[(size_t)m * sh.N + n] = t;
                }
            for (int ki = 0; ki < kNumGemmKinds; ++ki) {
                GemmKind kind = (GemmKind)ki;
                CUDA_CHECK(cudaMemset(dC, 0xff, (size_t)sh.M * sh.N * sizeof(float)));  // NaN poison
                gemm_bias_act(kind, dA, dW, db, dC, sh.M, sh.N, sh.K, act, s);
                CUDA_CHECK(cudaDeviceSynchronize());
                auto C = to_host(dC, (size_t)sh.M * sh.N);
                bool ok = true;
                float worst = 0;
                for (size_t i = 0; i < C.size(); ++i) {
                    float e = std::fabs(C[i] - ref[i]);
                    if (!(e <= 2e-6f * mag[i] + 1e-6f)) ok = false;  // also catches NaN
                    if (e > worst || std::isnan(e)) worst = e;
                }
                char name[128];
                std::snprintf(name, sizeof name, "gemm %-6s act=%d M=%d N=%d K=%d (max err %.2e)", gemm_name(kind), act,
                              sh.M, sh.N, sh.K, worst);
                report(ok, name);
            }
        }
        cudaFree(dA), cudaFree(dW), cudaFree(db), cudaFree(dC);
    }
}

static void test_gemm_fixed(std::mt19937& rng) {
    const FixedFormat formats[] = {{8, 7}, {15, 13}, {16, 12}, {20, 16}, {32, 16}};
    struct Shape { int M, N, K; };
    const Shape shapes[] = {{32, 512, 128}, {32, 128, 512}, {5, 37, 70}};
    for (const FixedFormat& f : formats)
        for (const Shape& sh : shapes) {
            FixedLayer cfg;
            cfg.enabled = true;
            cfg.data = cfg.weight = f;
            auto A = random_vec((size_t)sh.M * sh.K, -2, 2, rng);
            auto W = random_vec((size_t)sh.K * sh.N, -0.5f, 0.5f, rng);
            auto b = random_vec(sh.N, -1, 1, rng);
            float *dA = to_device(A), *dW = to_device(W), *db = to_device(b), *dC;
            CUDA_CHECK(cudaMalloc(&dC, (size_t)sh.M * sh.N * sizeof(float)));
            std::vector<float> ref((size_t)sh.M * sh.N);
            gemm_bias_fixed_cpu(A.data(), W.data(), b.data(), ref.data(), sh.M, sh.N, sh.K, cfg, kActNone);
            gemm_bias_fixed(dA, dW, db, dC, sh.M, sh.N, sh.K, cfg, kActNone, 0);
            CUDA_CHECK(cudaDeviceSynchronize());
            auto C = to_host(dC, ref.size());
            int mismatches = 0;
            for (size_t i = 0; i < C.size(); ++i) mismatches += C[i] != ref[i];
            char name[128];
            std::snprintf(name, sizeof name, "fixed WL=%d FL=%d M=%d N=%d K=%d bit-exact (%d mismatches)", f.wl, f.fl,
                          sh.M, sh.N, sh.K, mismatches);
            report(mismatches == 0, name);
            cudaFree(dA), cudaFree(dW), cudaFree(db), cudaFree(dC);
        }
}

static void test_attention(std::mt19937& rng) {
    const int batch = 3, seq = kMaxSeq, heads = kHeads, hd = kHeadDim, D = heads * hd;
    const int masks[batch] = {10, 32, 1};
    const size_t n = (size_t)batch * seq * D;
    auto q = random_vec(n, -1, 1, rng), k = random_vec(n, -1, 1, rng), v = random_vec(n, -1, 1, rng);
    float *dq = to_device(q), *dk = to_device(k), *dv = to_device(v), *dout;
    int* dm;
    CUDA_CHECK(cudaMalloc(&dout, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dm, sizeof masks));
    CUDA_CHECK(cudaMemcpy(dm, masks, sizeof masks, cudaMemcpyHostToDevice));
    attention(dq, dk, dv, dout, batch, seq, heads, hd, dm, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto out = to_host(dout, n);
    for (int b = 0; b < batch; ++b) {
        std::vector<float> ref((size_t)seq * D);
        const size_t off = (size_t)b * seq * D;
        attention_cpu(q.data() + off, k.data() + off, v.data() + off, ref.data(), seq, heads, hd, masks[b]);
        float worst = 0;
        for (size_t i = 0; i < ref.size(); ++i) worst = std::fmax(worst, std::fabs(out[off + i] - ref[i]));
        char name[96];
        std::snprintf(name, sizeof name, "attention sentence %d mask=%d (max err %.2e)", b, masks[b], worst);
        report(worst < 1e-5f, name);
    }
    cudaFree(dq), cudaFree(dk), cudaFree(dv), cudaFree(dout), cudaFree(dm);
}

int main() {
    std::mt19937 rng(4868);
    std::printf("FP32 GEMM\n");
    test_gemm_fp32(rng);
    std::printf("Fixed-point GEMM\n");
    test_gemm_fixed(rng);
    std::printf("Attention\n");
    test_attention(rng);
    cublas_shutdown();
    std::printf("\n%d/%d checks passed\n", g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
