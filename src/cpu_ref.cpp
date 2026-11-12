#include "cpu_ref.h"

#include <cmath>
#include <cstdio>

namespace bert {

const char* const kSentences[kNumSentences] = {"love", "happy", "great", "thoughtful", "confident",
                                               "slow", "boring", "noisy", "cringe", "hollow"};
const int kMasks[kNumSentences] = {10, 19, 21, 11, 8, 19, 22, 10, 13, 12};
const int kLabels[kNumSentences] = {1, 1, 1, 1, 1, 0, 0, 0, 0, 0};

FixedConfig soc_fixed_config() {
    FixedConfig c;
    c.ffn1.enabled = true;
    c.ffn1.data = {8, 7};
    c.ffn1.weight = {8, 7};
    c.ffn2.enabled = true;
    c.ffn2.data = {15, 13};
    c.ffn2.weight = {15, 13};
    return c;
}

static bool read_file(const std::string& path, std::vector<float>* out, size_t count, std::string* err) {
    out->assign(count, 0.0f);
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        if (err) *err = "cannot open " + path;
        return false;
    }
    size_t got = std::fread(out->data(), sizeof(float), count, f);
    std::fclose(f);
    if (got != count) {
        if (err) *err = path + ": expected " + std::to_string(count) + " floats, read " + std::to_string(got);
        return false;
    }
    return true;
}

bool load_weights(const std::string& dir, Weights* w, std::string* err) {
    const std::string d = dir.empty() || dir.back() == '/' ? dir : dir + "/";
    const size_t DD = (size_t)kDModel * kDModel;
    return read_file(d + "query_w_t.bin", &w->qw, DD, err) && read_file(d + "query_b.bin", &w->qb, kDModel, err) &&
           read_file(d + "key_w_t.bin", &w->kw, DD, err) && read_file(d + "key_b.bin", &w->kb, kDModel, err) &&
           read_file(d + "value_w_t.bin", &w->vw, DD, err) && read_file(d + "value_b.bin", &w->vb, kDModel, err) &&
           read_file(d + "dense_w_t.bin", &w->dw, DD, err) && read_file(d + "dense_b.bin", &w->db, kDModel, err) &&
           read_file(d + "ffn1_w_t.bin", &w->f1w, (size_t)kDModel * kFfn, err) &&
           read_file(d + "ffn1_b.bin", &w->f1b, kFfn, err) &&
           read_file(d + "ffn2_w_t.bin", &w->f2w, (size_t)kFfn * kDModel, err) &&
           read_file(d + "ffn2_b.bin", &w->f2b, kDModel, err) &&
           read_file(d + "pooler_w_t.bin", &w->pw, DD, err) && read_file(d + "pooler_b.bin", &w->pb, kDModel, err) &&
           read_file(d + "classifier_w_t.bin", &w->cw, (size_t)kDModel * kClasses, err) &&
           read_file(d + "classifier_b.bin", &w->cb, kClasses, err);
}

bool load_sentence(const std::string& dir, int idx, std::vector<float>* x, std::string* err) {
    const std::string d = dir.empty() || dir.back() == '/' ? dir : dir + "/";
    return read_file(d + kSentences[idx] + ".bin", x, (size_t)kMaxSeq * kDModel, err);
}

float gelu(float x) {
    const double kPi = 3.14159265358979323846;
    return (float)(0.5 * x * (1 + tanhf((float)(std::sqrt(2.0 / kPi) * (x + 0.044715 * x * x * x)))));
}

static float apply_act(float v, int act) {
    if (act == 1) return gelu(v);
    if (act == 2) return tanhf(v);
    return v;
}

void gemm_bias_cpu(const float* A, const float* W, const float* bias, float* C, int M, int N, int K, int act) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;  // the course model accumulated into un-initialized malloc() memory
            for (int k = 0; k < K; ++k) acc += W[(size_t)k * N + n] * A[(size_t)m * K + k];
            C[(size_t)m * N + n] = apply_act(acc + bias[n], act);
        }
}

void gemm_bias_fixed_cpu(const float* A, const float* W, const float* bias, float* C, int M, int N, int K,
                         const FixedLayer& cfg, int act) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            C[(size_t)m * N + n] = apply_act(fx_dot_bias(A + (size_t)m * K, W, N, n, bias[n], K, cfg), act);
}

void attention_cpu(const float* q, const float* k, const float* v, float* out, int seq, int heads, int head_dim,
                   int mask) {
    const int D = heads * head_dim;
    std::vector<float> p(seq);
    for (int h = 0; h < heads; ++h)
        for (int s = 0; s < seq; ++s) {
            float sum = 0.0f;
            for (int t = 0; t < seq; ++t) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) score += q[s * D + h * head_dim + d] * k[t * D + h * head_dim + d];
                // Same (unusual) softmax as the course model: exp(1 + score / 8), masked keys dropped.
                float e = expf(1.0f + score / 8.0f);
                p[t] = t < mask ? e : 0.0f;
                if (t < mask) sum += e;
            }
            for (int d = 0; d < head_dim; ++d) {
                float acc = 0.0f;
                for (int t = 0; t < seq; ++t) acc += (p[t] / sum) * v[t * D + h * head_dim + d];
                out[s * D + h * head_dim + d] = acc;
            }
        }
}

void forward_cpu(const Weights& w, const float* x, int mask, const FixedConfig& fx, float logits[kClasses]) {
    const int S = kMaxSeq, D = kDModel, F = kFfn;
    std::vector<float> q(S * D), k(S * D), v(S * D), attn(S * D), dense(S * D), r1(S * D);
    std::vector<float> f1(S * F), f2(S * D), r2(S * D), pooled(D);

    gemm_bias_cpu(x, w.qw.data(), w.qb.data(), q.data(), S, D, D, 0);
    gemm_bias_cpu(x, w.kw.data(), w.kb.data(), k.data(), S, D, D, 0);
    gemm_bias_cpu(x, w.vw.data(), w.vb.data(), v.data(), S, D, D, 0);
    attention_cpu(q.data(), k.data(), v.data(), attn.data(), S, kHeads, kHeadDim, mask);
    gemm_bias_cpu(attn.data(), w.dw.data(), w.db.data(), dense.data(), S, D, D, 0);
    for (int i = 0; i < S * D; ++i) r1[i] = x[i] + dense[i];

    if (fx.ffn1.enabled)
        gemm_bias_fixed_cpu(r1.data(), w.f1w.data(), w.f1b.data(), f1.data(), S, F, D, fx.ffn1, 1);
    else
        gemm_bias_cpu(r1.data(), w.f1w.data(), w.f1b.data(), f1.data(), S, F, D, 1);
    if (fx.ffn2.enabled)
        gemm_bias_fixed_cpu(f1.data(), w.f2w.data(), w.f2b.data(), f2.data(), S, D, F, fx.ffn2, 0);
    else
        gemm_bias_cpu(f1.data(), w.f2w.data(), w.f2b.data(), f2.data(), S, D, F, 0);
    for (int i = 0; i < S * D; ++i) r2[i] = r1[i] + f2[i];

    // Pooler and classifier use the first ([CLS]) token only.
    gemm_bias_cpu(r2.data(), w.pw.data(), w.pb.data(), pooled.data(), 1, D, D, 2);
    gemm_bias_cpu(pooled.data(), w.cw.data(), w.cb.data(), logits, 1, kClasses, D, 0);
}

}  // namespace bert
