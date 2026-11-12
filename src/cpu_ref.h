// CPU reference for the single-block BERT sentiment classifier from the
// CSEE 4868 SoC project, plus loaders for its binary weight/sentence files.
#pragma once

#include <string>
#include <vector>

#include "fixed_point.h"

namespace bert {

constexpr int kMaxSeq = 32;
constexpr int kDModel = 128;
constexpr int kHeads = 2;
constexpr int kHeadDim = kDModel / kHeads;
constexpr int kFfn = 4 * kDModel;
constexpr int kClasses = 2;
constexpr int kNumSentences = 10;

extern const char* const kSentences[kNumSentences];
extern const int kMasks[kNumSentences];   // number of real tokens per sentence
extern const int kLabels[kNumSentences];  // 1 = positive, 0 = negative

// Weight matrices are stored [in][out] row-major, as in the *_w_t.bin files.
struct Weights {
    std::vector<float> qw, qb, kw, kb, vw, vb, dw, db;
    std::vector<float> f1w, f1b, f2w, f2b, pw, pb, cw, cb;
};

struct FixedConfig {
    FixedLayer ffn1, ffn2;
};

// Formats used by the team's SoC application (systest.c): FFN1 8-bit with
// 1 integer bit, FFN2 15-bit with 2 integer bits; weights use the same format.
FixedConfig soc_fixed_config();

bool load_weights(const std::string& dir, Weights* w, std::string* err);
bool load_sentence(const std::string& dir, int idx, std::vector<float>* x, std::string* err);

float gelu(float x);

// C[M][N] = act(A[M][K] * W[K][N] + bias[N]); act: 0 none, 1 GELU, 2 tanh
void gemm_bias_cpu(const float* A, const float* W, const float* bias, float* C,
                   int M, int N, int K, int act);
void gemm_bias_fixed_cpu(const float* A, const float* W, const float* bias, float* C,
                         int M, int N, int K, const FixedLayer& cfg, int act);

// q/k/v/out: [seq][heads*head_dim]; keys at positions >= mask are ignored.
void attention_cpu(const float* q, const float* k, const float* v, float* out,
                   int seq, int heads, int head_dim, int mask);

void forward_cpu(const Weights& w, const float* x, int mask, const FixedConfig& fx,
                 float logits[kClasses]);

}  // namespace bert
