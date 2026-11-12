#include "bert_gpu.cuh"

#include "cuda_check.h"

namespace bert {

GpuBert::GpuBert(const Weights& w, int max_batch, cudaStream_t stream) : s_(stream), max_batch_(max_batch) {
    auto upload_vec = [&](const std::vector<float>& h) {
        float* d = nullptr;
        CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
        allocs_.push_back(d);
        return d;
    };
    auto alloc = [&](size_t n) {
        float* d = nullptr;
        CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
        CUDA_CHECK(cudaMemset(d, 0, n * sizeof(float)));
        allocs_.push_back(d);
        return d;
    };
    qw_ = upload_vec(w.qw), qb_ = upload_vec(w.qb), kw_ = upload_vec(w.kw), kb_ = upload_vec(w.kb);
    vw_ = upload_vec(w.vw), vb_ = upload_vec(w.vb), dw_ = upload_vec(w.dw), db_ = upload_vec(w.db);
    f1w_ = upload_vec(w.f1w), f1b_ = upload_vec(w.f1b), f2w_ = upload_vec(w.f2w), f2b_ = upload_vec(w.f2b);
    pw_ = upload_vec(w.pw), pb_ = upload_vec(w.pb), cw_ = upload_vec(w.cw), cb_ = upload_vec(w.cb);

    const size_t T = (size_t)max_batch * kMaxSeq;  // tokens
    x_ = alloc(T * kDModel), q_ = alloc(T * kDModel), k_ = alloc(T * kDModel), v_ = alloc(T * kDModel);
    attn_ = alloc(T * kDModel), dense_ = alloc(T * kDModel), r1_ = alloc(T * kDModel);
    f1_ = alloc(T * kFfn), f2_ = alloc(T * kDModel), r2_ = alloc(T * kDModel);
    cls_ = alloc((size_t)max_batch * kDModel), pooled_ = alloc((size_t)max_batch * kDModel);
    logits_ = alloc((size_t)max_batch * kClasses);
    CUDA_CHECK(cudaMalloc(&masks_, max_batch * sizeof(int)));
    allocs_.push_back(masks_);

    cublas_init(s_);  // must happen before any graph capture
}

GpuBert::~GpuBert() {
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    if (graph_) cudaGraphDestroy(graph_);
    for (void* p : allocs_) cudaFree(p);
}

void GpuBert::upload(const float* h_x, const int* masks, int batch) {
    CUDA_CHECK(cudaMemcpyAsync(x_, h_x, sizeof(float) * batch * kMaxSeq * kDModel, cudaMemcpyHostToDevice, s_));
    CUDA_CHECK(cudaMemcpyAsync(masks_, masks, sizeof(int) * batch, cudaMemcpyHostToDevice, s_));
}

void GpuBert::linear(const float* in, const float* w, const float* b, float* out, int M, int N, int K, int act,
                     const FixedLayer* fx) {
    if (fx && fx->enabled)
        gemm_bias_fixed(in, w, b, out, M, N, K, *fx, act, s_);
    else
        gemm_bias_act(gemm_, in, w, b, out, M, N, K, act, s_);
}

void GpuBert::ffn(int batch) {
    const int T = batch * kMaxSeq;
    linear(r1_, f1w_, f1b_, f1_, T, kFfn, kDModel, kActGelu, &fx_.ffn1);
    linear(f1_, f2w_, f2b_, f2_, T, kDModel, kFfn, kActNone, &fx_.ffn2);
}

void GpuBert::forward(int batch) {
    const int T = batch * kMaxSeq, D = kDModel;
    // Q, K, V projections over all tokens of all sentences at once.
    linear(x_, qw_, qb_, q_, T, D, D, kActNone, nullptr);
    linear(x_, kw_, kb_, k_, T, D, D, kActNone, nullptr);
    linear(x_, vw_, vb_, v_, T, D, D, kActNone, nullptr);
    attention(q_, k_, v_, attn_, batch, kMaxSeq, kHeads, kHeadDim, masks_, s_);
    linear(attn_, dw_, db_, dense_, T, D, D, kActNone, nullptr);
    add(x_, dense_, r1_, T * D, s_);
    ffn(batch);  // FFN1 (+GELU) and FFN2
    add(r1_, f2_, r2_, T * D, s_);
    gather_rows(r2_, cls_, batch, kMaxSeq, D, s_);  // [CLS] token of each sentence
    linear(cls_, pw_, pb_, pooled_, batch, D, D, kActTanh, nullptr);
    linear(pooled_, cw_, cb_, logits_, batch, kClasses, D, kActNone, nullptr);
}

void GpuBert::download_logits(float* h_logits, int batch) {
    CUDA_CHECK(cudaMemcpyAsync(h_logits, logits_, sizeof(float) * batch * kClasses, cudaMemcpyDeviceToHost, s_));
    CUDA_CHECK(cudaStreamSynchronize(s_));
}

void GpuBert::capture_graph(int batch) {
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_), graph_exec_ = nullptr;
    if (graph_) cudaGraphDestroy(graph_), graph_ = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(s_, cudaStreamCaptureModeThreadLocal));
    forward(batch);
    CUDA_CHECK(cudaStreamEndCapture(s_, &graph_));
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph_exec_, graph_, 0));
}

void GpuBert::launch_graph() { CUDA_CHECK(cudaGraphLaunch(graph_exec_, s_)); }

}  // namespace bert
