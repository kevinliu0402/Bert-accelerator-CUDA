// perf_gemm: one GEMM kernel per run, with everything needed to connect the
// compiler's output to what the GPU did:
//   registers, spills (local memory), static shared memory   <- cudaFuncGetAttributes
//   resident blocks per SM, theoretical occupancy, waves     <- occupancy API
//   time (median of reps), GFLOP/s, max relative error vs CPU
//
//   perf_gemm --kernel NAME|all|main  (main = everything except the reg64_minb* sweep)
//             [--M 2048 --N 512 --K 128] [--iters 200] [--reps 5]
//             [--warmup 10] [--warm-ms 300] [--tag T] [--csv FILE]
//
// --warm-ms: after the counted warm-up, keep launching for this long before timing.
// The first Colab run showed why: the first kernel timed after the ~1 s CPU
// reference had erratic times (same SASS, 160-227 us), consistent with the GPU
// clock still ramping up. Counted warm-up of 10 launches is only ~1 ms.
//
// Under Nsight Compute use: --iters 1 --reps 1 --warmup 2 --warm-ms 0, then profile with
// --launch-skip 2 --launch-count 1, so ncu sees the first timed launch.
//
// It includes kernels.cu directly (one translation unit) so it can take the
// address of every kernel template instance, including the PAD variants.
#include "kernels.cu"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace bert;

namespace {

using Launch = void (*)(const float*, const float*, const float*, float*, int, int, int, cudaStream_t);

struct KernelInfo {
    const char* name;
    const void* func;  // nullptr for cuBLAS (closed source: no attributes)
    int threads;       // per block
    int bm, bn;        // output tile per block
    Launch launch;
};

template <int BM, int BN, int BK, int TM, int TN, int PAD, int MINB = 1>
void launch_rb(const float* A, const float* W, const float* b, float* C, int M, int N, int K, cudaStream_t s) {
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_regblock_kernel<BM, BN, BK, TM, TN, PAD, MINB><<<grid, (BM / TM) * (BN / TN), 0, s>>>(A, W, b, C, M, N, K, kActNone);
}
void launch_kind(GemmKind k, const float* A, const float* W, const float* b, float* C, int M, int N, int K,
                 cudaStream_t s) {
    gemm_bias_act(k, A, W, b, C, M, N, K, kActNone, s);
}
void l_naive(const float* A, const float* W, const float* b, float* C, int M, int N, int K, cudaStream_t s) {
    launch_kind(GemmKind::Naive, A, W, b, C, M, N, K, s);
}
void l_tiled(const float* A, const float* W, const float* b, float* C, int M, int N, int K, cudaStream_t s) {
    launch_kind(GemmKind::Tiled, A, W, b, C, M, N, K, s);
}
void l_cublas(const float* A, const float* W, const float* b, float* C, int M, int N, int K, cudaStream_t s) {
    launch_kind(GemmKind::Cublas, A, W, b, C, M, N, K, s);
}

const KernelInfo kKernels[] = {
    {"naive", (const void*)gemm_naive_kernel, 256, 8, 32, l_naive},
    {"tiled", (const void*)gemm_tiled_kernel, 1024, 32, 32, l_tiled},
    {"reg32", (const void*)gemm_regblock_kernel<32, 32, 32, 2, 2, 1>, 256, 32, 32, launch_rb<32, 32, 32, 2, 2, 1>},
    {"reg64", (const void*)gemm_regblock_kernel<64, 64, 16, 4, 4, 1>, 256, 64, 64, launch_rb<64, 64, 16, 4, 4, 1>},
    {"reg64_pad0", (const void*)gemm_regblock_kernel<64, 64, 16, 4, 4, 0>, 256, 64, 64,
     launch_rb<64, 64, 16, 4, 4, 0>},
    {"reg64_pad4", (const void*)gemm_regblock_kernel<64, 64, 16, 4, 4, 4>, 256, 64, 64,
     launch_rb<64, 64, 16, 4, 4, 4>},
    // Register-budget knob without rebuilding: __launch_bounds__(256, MINB) caps
    // registers at 65536 / (MINB * 256) = 128 / 85 / 64 for MINB = 2 / 3 / 4.
    // (Turing allows at most 1024 threads = 4 such blocks per SM.)
    {"reg64_minb2", (const void*)gemm_regblock_kernel<64, 64, 16, 4, 4, 1, 2>, 256, 64, 64,
     launch_rb<64, 64, 16, 4, 4, 1, 2>},
    {"reg64_minb3", (const void*)gemm_regblock_kernel<64, 64, 16, 4, 4, 1, 3>, 256, 64, 64,
     launch_rb<64, 64, 16, 4, 4, 1, 3>},
    {"reg64_minb4", (const void*)gemm_regblock_kernel<64, 64, 16, 4, 4, 1, 4>, 256, 64, 64,
     launch_rb<64, 64, 16, 4, 4, 1, 4>},
    {"cublas", nullptr, 0, 0, 0, l_cublas},
};

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

}  // namespace

int main(int argc, char** argv) {
    std::string which = "all", tag = "default", csv;
    int M = 2048, N = 512, K = 128, iters = 200, reps = 5, warmup = 10;
    double warm_ms = 300;
    for (int i = 1; i < argc; ++i) {
        auto next = [&](void) -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (!std::strcmp(argv[i], "--kernel")) which = next();
        else if (!std::strcmp(argv[i], "--M")) M = std::atoi(next());
        else if (!std::strcmp(argv[i], "--N")) N = std::atoi(next());
        else if (!std::strcmp(argv[i], "--K")) K = std::atoi(next());
        else if (!std::strcmp(argv[i], "--iters")) iters = std::atoi(next());
        else if (!std::strcmp(argv[i], "--reps")) reps = std::atoi(next());
        else if (!std::strcmp(argv[i], "--warmup")) warmup = std::atoi(next());
        else if (!std::strcmp(argv[i], "--warm-ms")) warm_ms = std::atof(next());
        else if (!std::strcmp(argv[i], "--tag")) tag = next();
        else if (!std::strcmp(argv[i], "--csv")) csv = next();
        else {
            std::fprintf(stderr, "unknown argument %s\n", argv[i]);
            return 2;
        }
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    cublas_init(s);

    // Inputs and a CPU reference (double accumulation).
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> u(-1, 1);
    std::vector<float> hA((size_t)M * K), hW((size_t)K * N), hb(N), hC((size_t)M * N);
    for (auto& x : hA) x = u(rng);
    for (auto& x : hW) x = 0.1f * u(rng);
    for (auto& x : hb) x = u(rng);
    std::vector<double> ref((size_t)M * N);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double acc = hb[n];
            for (int k = 0; k < K; ++k) acc += (double)hA[(size_t)m * K + k] * hW[(size_t)k * N + n];
            ref[(size_t)m * N + n] = acc;
        }

    float *A, *W, *b, *C;
    CUDA_CHECK(cudaMalloc(&A, hA.size() * 4));
    CUDA_CHECK(cudaMalloc(&W, hW.size() * 4));
    CUDA_CHECK(cudaMalloc(&b, hb.size() * 4));
    CUDA_CHECK(cudaMalloc(&C, hC.size() * 4));
    CUDA_CHECK(cudaMemcpy(A, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(W, hW.data(), hW.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice));

    FILE* f = nullptr;
    if (!csv.empty()) {
        FILE* probe = std::fopen(csv.c_str(), "r");
        bool fresh = probe == nullptr;
        if (probe) std::fclose(probe);
        f = std::fopen(csv.c_str(), "a");
        if (fresh)
            std::fprintf(f,
                         "gpu,tag,kernel,M,N,K,threads,blocks,regs,local_bytes,static_smem,blocks_per_sm,"
                         "occupancy_pct,waves,us_median,us_min,cv_pct,gflops,max_rel_err\n");
    }

    std::printf("GPU %s, %d SMs, %d regs/SM, %d max threads/SM | M=%d N=%d K=%d | tag=%s\n", prop.name,
                prop.multiProcessorCount, prop.regsPerMultiprocessor, prop.maxThreadsPerMultiProcessor, M, N, K,
                tag.c_str());
    GpuTimer timer;
    const double flops = 2.0 * M * N * K;
    int ran = 0;
    for (const KernelInfo& k : kKernels) {
        bool is_sweep = std::strncmp(k.name, "reg64_minb", 10) == 0;
        if (which == "main" ? is_sweep : (which != "all" && which != k.name)) continue;
        ++ran;
        int regs = -1, local = -1, smem = -1, per_sm = -1;
        long blocks = -1;
        double occ = -1, waves = -1;
        if (k.func) {
            cudaFuncAttributes at;
            CUDA_CHECK(cudaFuncGetAttributes(&at, k.func));
            regs = at.numRegs;
            local = (int)at.localSizeBytes;  // > 0 means the compiler spilled registers
            smem = (int)at.sharedSizeBytes;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k.func, k.threads, 0));
            blocks = (long)((M + k.bm - 1) / k.bm) * ((N + k.bn - 1) / k.bn);
            occ = 100.0 * per_sm * k.threads / prop.maxThreadsPerMultiProcessor;
            waves = (double)blocks / ((double)per_sm * prop.multiProcessorCount);
        }

        for (int i = 0; i < warmup; ++i) k.launch(A, W, b, C, M, N, K, s);
        CUDA_CHECK(cudaStreamSynchronize(s));
        for (double spent = 0; spent < warm_ms;) {  // timed warm-up: let clocks settle
            timer.start(s);
            for (int i = 0; i < 50; ++i) k.launch(A, W, b, C, M, N, K, s);
            spent += timer.stop_ms(s);
        }
        std::vector<double> us;
        for (int r = 0; r < reps; ++r) {
            timer.start(s);
            for (int i = 0; i < iters; ++i) k.launch(A, W, b, C, M, N, K, s);
            us.push_back(1000.0 * timer.stop_ms(s) / iters);
        }
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(hC.data(), C, hC.size() * 4, cudaMemcpyDeviceToHost));
        double err = 0;
        for (size_t i = 0; i < hC.size(); ++i)
            err = std::max(err, std::fabs(hC[i] - ref[i]) / std::max(1.0, std::fabs(ref[i])));

        double med = median(us), mn = *std::min_element(us.begin(), us.end()), mean = 0, var = 0;
        for (double x : us) mean += x / us.size();
        for (double x : us) var += (x - mean) * (x - mean) / us.size();
        double cv = mean > 0 ? 100.0 * std::sqrt(var) / mean : 0;
        double gf = flops / (med * 1e3);

        std::printf("  %-11s %8.2f us %8.1f GFLOP/s  regs %3d  spill %4d B  smem %5d B  %2d blk/SM (%5.1f%% occ)"
                    "  %6.2f waves  cv %4.1f%%  err %.1e%s\n",
                    k.name, med, gf, regs, local, smem, per_sm, occ, waves, cv, err, err > 1e-3 ? "  FAIL" : "");
        if (f)
            std::fprintf(f, "\"%s\",%s,%s,%d,%d,%d,%d,%ld,%d,%d,%d,%d,%.1f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2e\n",
                         prop.name, tag.c_str(), k.name, M, N, K, k.threads, blocks, regs, local, smem, per_sm, occ,
                         waves, med, mn, cv, gf, err);
    }
    if (!ran) {
        std::fprintf(stderr, "no kernel named %s\n", which.c_str());
        return 2;
    }
    if (f) std::fclose(f);
    cudaFree(A), cudaFree(W), cudaFree(b), cudaFree(C);
    cublas_shutdown();
    return 0;
}
