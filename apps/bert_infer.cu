// Runs the project's BERT sentiment classifier on the GPU and compares it with
// the CPU reference.
//
//   bert_infer --data DIR [--gemm all|naive|tiled|reg32|reg64|cublas]
//              [--fixed WL,FL] [--sweep] [--iters N] [--csv FILE]
//
// For each GEMM implementation it reports:
//   * accuracy on the 10 sentences and the max |logit| error vs. the CPU model
//   * latency of one sentence: kernels launched one by one vs. one CUDA graph
//   * latency of the FFN layers alone (what the SoC accelerators computed)
//   * throughput with all 10 sentences in one batch
// --fixed runs FFN1/FFN2 with the project's fixed-point arithmetic.
// --sweep tries many fixed-point formats and reports accuracy for each.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "bert_gpu.cuh"
#include "cpu_ref.h"
#include "cuda_check.h"

using namespace bert;

struct Options {
    std::string data = "data";
    std::string gemm = "all";
    std::string csv;
    bool fixed = false;
    FixedFormat fx{32, 16};
    bool sweep = false;
    int iters = 200;
};

static void usage() {
    std::fprintf(stderr,
                 "usage: bert_infer --data DIR [--gemm all|naive|tiled|reg32|reg64|cublas]\n"
                 "                  [--fixed WL,FL] [--sweep] [--iters N] [--csv FILE]\n");
}

static bool parse(int argc, char** argv, Options* o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        const char* v = nullptr;
        if (a == "--data" && (v = next())) o->data = v;
        else if (a == "--gemm" && (v = next())) o->gemm = v;
        else if (a == "--csv" && (v = next())) o->csv = v;
        else if (a == "--iters" && (v = next())) o->iters = std::atoi(v);
        else if (a == "--sweep") o->sweep = true;
        else if (a == "--fixed" && (v = next())) {
            o->fixed = true;
            if (std::sscanf(v, "%d,%d", &o->fx.wl, &o->fx.fl) != 2) return false;
        } else return false;
    }
    return o->iters > 0;
}

static FixedConfig make_fixed(const FixedFormat& f) {
    FixedConfig c;
    c.ffn1.enabled = c.ffn2.enabled = true;
    c.ffn1.data = c.ffn1.weight = c.ffn2.data = c.ffn2.weight = f;
    return c;
}

struct Result {
    int correct = 0;
    float max_err = 0;
    float eager_us = 0, graph_us = 0, ffn_us = 0, batch_us = 0;
};

int main(int argc, char** argv) {
    Options opt;
    if (!parse(argc, argv, &opt)) {
        usage();
        return 2;
    }

    Weights w;
    std::string err;
    if (!load_weights(opt.data, &w, &err)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 2;
    }
    std::vector<float> xs((size_t)kNumSentences * kMaxSeq * kDModel);
    for (int i = 0; i < kNumSentences; ++i) {
        std::vector<float> x;
        if (!load_sentence(opt.data, i, &x, &err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 2;
        }
        std::copy(x.begin(), x.end(), xs.begin() + (size_t)i * kMaxSeq * kDModel);
    }
    auto sentence = [&](int i) { return xs.data() + (size_t)i * kMaxSeq * kDModel; };

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s (%d SMs, sm_%d%d)\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    GpuBert model(w, kNumSentences, s);

    // ---------------------------------------------------------------- sweep
    if (opt.sweep) {
        std::printf("\nFixed-point sweep of FFN1/FFN2 (data and weights share one format)\n");
        std::printf("%4s %4s %4s %10s %14s\n", "WL", "IL", "FL", "accuracy", "max|logit err|");
        for (int wl : {8, 10, 12, 14, 16, 18, 20, 24, 32})
            for (int il : {1, 2, 3, 4, 6, 8}) {
                if (il >= wl) continue;
                FixedConfig fx = make_fixed({wl, wl - il});
                model.set_fixed(fx);
                model.upload(xs.data(), kMasks, kNumSentences);
                model.forward(kNumSentences);
                float logits[kNumSentences * kClasses];
                model.download_logits(logits, kNumSentences);
                int ok = 0;
                float e = 0;
                for (int i = 0; i < kNumSentences; ++i) {
                    float ref[2];
                    forward_cpu(w, sentence(i), kMasks[i], fx, ref);
                    ok += (logits[2 * i + 1] > logits[2 * i]) == kLabels[i];
                    e = std::fmax(e, std::fmax(std::fabs(ref[0] - logits[2 * i]), std::fabs(ref[1] - logits[2 * i + 1])));
                }
                std::printf("%4d %4d %4d %7d/10 %14.2e\n", wl, il, wl - il, ok, e);
            }
        return 0;
    }

    // ------------------------------------------------------- per-GEMM runs
    const FixedConfig fx = opt.fixed ? make_fixed(opt.fx) : FixedConfig();
    model.set_fixed(fx);
    float ref_logits[kNumSentences][kClasses];
    int ref_correct = 0;
    for (int i = 0; i < kNumSentences; ++i) {
        forward_cpu(w, sentence(i), kMasks[i], fx, ref_logits[i]);
        ref_correct += (ref_logits[i][1] > ref_logits[i][0]) == kLabels[i];
    }
    std::printf("FFN arithmetic: %s\n", opt.fixed ? ("fixed point " + std::to_string(opt.fx.wl) + "," +
                                                     std::to_string(opt.fx.fl)).c_str()
                                                  : "FP32");
    std::printf("CPU reference accuracy: %d/10\n\n", ref_correct);

    std::vector<GemmKind> kinds;
    if (opt.gemm == "all") {
        for (int i = 0; i < kNumGemmKinds; ++i) kinds.push_back((GemmKind)i);
    } else {
        GemmKind k;
        if (!parse_gemm(opt.gemm.c_str(), &k)) {
            usage();
            return 2;
        }
        kinds.push_back(k);
    }

    GpuTimer timer;
    std::vector<Result> results;
    for (GemmKind kind : kinds) {
        Result r;
        model.set_gemm(kind);

        // Correctness: each sentence on its own (batch = 1).
        for (int i = 0; i < kNumSentences; ++i) {
            float lg[2];
            model.upload(sentence(i), &kMasks[i], 1);
            model.forward(1);
            model.download_logits(lg, 1);
            r.correct += (lg[1] > lg[0]) == kLabels[i];
            for (int c = 0; c < 2; ++c) r.max_err = std::fmax(r.max_err, std::fabs(lg[c] - ref_logits[i][c]));
        }

        // Latency of one sentence, kernels launched one by one.
        model.upload(sentence(0), &kMasks[0], 1);
        for (int i = 0; i < 10; ++i) model.forward(1);
        timer.start(s);
        for (int i = 0; i < opt.iters; ++i) model.forward(1);
        r.eager_us = 1000.0f * timer.stop_ms(s) / opt.iters;

        // Same work as one CUDA graph launch.
        model.capture_graph(1);
        for (int i = 0; i < 10; ++i) model.launch_graph();
        timer.start(s);
        for (int i = 0; i < opt.iters; ++i) model.launch_graph();
        r.graph_us = 1000.0f * timer.stop_ms(s) / opt.iters;

        // FFN1 + GELU + FFN2 only, for one sentence.
        timer.start(s);
        for (int i = 0; i < opt.iters; ++i) model.ffn(1);
        r.ffn_us = 1000.0f * timer.stop_ms(s) / opt.iters;

        // All 10 sentences in one batch, as a graph.
        model.upload(xs.data(), kMasks, kNumSentences);
        model.capture_graph(kNumSentences);
        for (int i = 0; i < 10; ++i) model.launch_graph();
        timer.start(s);
        for (int i = 0; i < opt.iters; ++i) model.launch_graph();
        r.batch_us = 1000.0f * timer.stop_ms(s) / opt.iters;
        float lg[kNumSentences * kClasses];
        model.download_logits(lg, kNumSentences);
        for (int i = 0; i < kNumSentences; ++i)
            for (int c = 0; c < 2; ++c) r.max_err = std::fmax(r.max_err, std::fabs(lg[2 * i + c] - ref_logits[i][c]));

        results.push_back(r);
    }

    std::printf("%-7s %8s %10s %14s %14s %12s %16s %14s\n", "gemm", "acc", "max err", "1 sent eager", "1 sent graph",
                "FFN only", "10-sent batch", "sentences/s");
    for (size_t i = 0; i < kinds.size(); ++i) {
        const Result& r = results[i];
        std::printf("%-7s %5d/10 %10.2e %11.1f us %11.1f us %9.1f us %13.1f us %14.0f\n", gemm_name(kinds[i]),
                    r.correct, r.max_err, r.eager_us, r.graph_us, r.ffn_us, r.batch_us,
                    kNumSentences * 1e6 / r.batch_us);
    }

    if (!opt.csv.empty()) {
        FILE* f = std::fopen(opt.csv.c_str(), "w");
        if (f) {
            std::fprintf(f, "gpu,ffn_arith,gemm,correct,max_err,eager_us,graph_us,ffn_us,batch10_us\n");
            for (size_t i = 0; i < kinds.size(); ++i) {
                const Result& r = results[i];
                std::fprintf(f, "\"%s\",%s,%s,%d,%g,%.2f,%.2f,%.2f,%.2f\n", prop.name, opt.fixed ? "fixed" : "fp32",
                             gemm_name(kinds[i]), r.correct, r.max_err, r.eager_us, r.graph_us, r.ffn_us, r.batch_us);
            }
            std::fclose(f);
            std::printf("\nwrote %s\n", opt.csv.c_str());
        }
    }

    // FP32: GPU and CPU differ only in rounding order and in device expf/tanhf.
    // Fixed point: those tiny FP32 differences in earlier layers can move a value
    // across a quantization step, so the logit tolerance is looser (the FFN kernels
    // themselves are checked bit-exact in test_kernels).
    const float tol = opt.fixed ? 5e-2f : 1e-3f;
    bool ok = true;
    for (const Result& r : results) ok = ok && r.correct == ref_correct && r.max_err < tol;
    cublas_shutdown();
    CUDA_CHECK(cudaStreamDestroy(s));
    std::printf("\n%s\n", ok ? "PASS: GPU matches the CPU reference" : "FAIL: GPU differs from the CPU reference");
    return ok ? 0 : 1;
}
