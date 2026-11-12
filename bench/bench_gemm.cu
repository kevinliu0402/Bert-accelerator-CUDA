// Benchmarks every GEMM implementation on the project's FFN shapes, from one
// sentence (32 tokens) up to large batches, and compares with cuBLAS.
//   bench_gemm [--iters N] [--csv FILE]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "cpu_ref.h"
#include "cuda_check.h"
#include "kernels.cuh"

using namespace bert;

int main(int argc, char** argv) {
    int iters = 200;
    std::string csv;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];
        else {
            std::fprintf(stderr, "usage: bench_gemm [--iters N] [--csv FILE]\n");
            return 2;
        }
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s (%d SMs)\n\n", prop.name, prop.multiProcessorCount);

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    cublas_init(s);
    GpuTimer timer;
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> u(-1, 1);

    FILE* f = csv.empty() ? nullptr : std::fopen(csv.c_str(), "w");
    if (f) std::fprintf(f, "gpu,layer,sentences,M,N,K,gemm,us,gflops,pct_cublas,blocks\n");

    struct Layer { const char* name; int N, K; };
    const Layer layers[] = {{"FFN1", kFfn, kDModel}, {"FFN2", kDModel, kFfn}};
    const int batches[] = {1, 10, 64, 256};

    for (const Layer& L : layers)
        for (int B : batches) {
            const int M = B * kMaxSeq, N = L.N, K = L.K;
            std::vector<float> hA((size_t)M * K), hW((size_t)K * N), hb(N);
            for (auto& x : hA) x = u(rng);
            for (auto& x : hW) x = 0.1f * u(rng);
            for (auto& x : hb) x = u(rng);
            float *A, *W, *b, *C;
            CUDA_CHECK(cudaMalloc(&A, hA.size() * 4));
            CUDA_CHECK(cudaMalloc(&W, hW.size() * 4));
            CUDA_CHECK(cudaMalloc(&b, hb.size() * 4));
            CUDA_CHECK(cudaMalloc(&C, (size_t)M * N * 4));
            CUDA_CHECK(cudaMemcpy(A, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(W, hW.data(), hW.size() * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(b, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice));

            const double flops = 2.0 * M * N * K;
            float us[kNumGemmKinds];
            for (int ki = 0; ki < kNumGemmKinds; ++ki) {
                GemmKind k = (GemmKind)ki;
                for (int i = 0; i < 10; ++i) gemm_bias_act(k, A, W, b, C, M, N, K, kActNone, s);
                timer.start(s);
                for (int i = 0; i < iters; ++i) gemm_bias_act(k, A, W, b, C, M, N, K, kActNone, s);
                us[ki] = 1000.0f * timer.stop_ms(s) / iters;
            }
            const float cublas_us = us[(int)GemmKind::Cublas];
            std::printf("%s  %3d sentence(s)  M=%-5d N=%-4d K=%-4d\n", L.name, B, M, N, K);
            for (int ki = 0; ki < kNumGemmKinds; ++ki) {
                GemmKind k = (GemmKind)ki;
                int bm = k == GemmKind::Reg64 ? 64 : (k == GemmKind::Naive ? 8 : 32);
                int bn = k == GemmKind::Reg64 ? 64 : 32;
                long blocks = k == GemmKind::Cublas ? -1 : (long)((M + bm - 1) / bm) * ((N + bn - 1) / bn);
                double gf = flops / (us[ki] * 1e3);
                std::printf("    %-7s %9.2f us %9.1f GFLOP/s %7.1f%% of cuBLAS", gemm_name(k), us[ki], gf,
                            100.0 * cublas_us / us[ki]);
                if (blocks > 0) std::printf("   (%ld blocks on %d SMs)", blocks, prop.multiProcessorCount);
                std::printf("\n");
                if (f)
                    std::fprintf(f, "\"%s\",%s,%d,%d,%d,%d,%s,%.3f,%.2f,%.2f,%ld\n", prop.name, L.name, B, M, N, K,
                                 gemm_name(k), us[ki], gf, 100.0 * cublas_us / us[ki], blocks);
            }
            std::printf("\n");
            cudaFree(A), cudaFree(W), cudaFree(b), cudaFree(C);
        }

    if (f) {
        std::fclose(f);
        std::printf("wrote %s\n", csv.c_str());
    }
    cublas_shutdown();
    CUDA_CHECK(cudaStreamDestroy(s));
    return 0;
}
