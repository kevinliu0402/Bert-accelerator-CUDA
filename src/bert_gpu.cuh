// The project's single-block BERT classifier on the GPU, for `batch` sentences at a time.
#pragma once

#include <vector>

#include "cpu_ref.h"
#include "kernels.cuh"

namespace bert {

class GpuBert {
  public:
    GpuBert(const Weights& w, int max_batch, cudaStream_t stream);
    ~GpuBert();
    GpuBert(const GpuBert&) = delete;
    GpuBert& operator=(const GpuBert&) = delete;

    void set_gemm(GemmKind k) { gemm_ = k; }
    void set_fixed(const FixedConfig& fx) { fx_ = fx; }

    // h_x: [batch][kMaxSeq][kDModel], masks: [batch]
    void upload(const float* h_x, const int* masks, int batch);
    // Enqueues the whole forward pass (12 kernels for any batch size).
    void forward(int batch);
    // Enqueues only FFN1 + GELU + FFN2 (the layers the SoC accelerated).
    void ffn(int batch);
    void download_logits(float* h_logits, int batch);

    // CUDA graph of forward(batch): removes per-kernel launch overhead.
    void capture_graph(int batch);
    void launch_graph();

  private:
    void linear(const float* in, const float* w, const float* b, float* out, int M, int N, int K, int act,
                const FixedLayer* fx);

    cudaStream_t s_;
    int max_batch_;
    GemmKind gemm_ = GemmKind::Reg32;
    FixedConfig fx_;
    // weights
    float *qw_, *qb_, *kw_, *kb_, *vw_, *vb_, *dw_, *db_, *f1w_, *f1b_, *f2w_, *f2b_, *pw_, *pb_, *cw_, *cb_;
    // activations
    float *x_, *q_, *k_, *v_, *attn_, *dense_, *r1_, *f1_, *f2_, *r2_, *cls_, *pooled_, *logits_;
    int* masks_;
    std::vector<void*> allocs_;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
};

}  // namespace bert
