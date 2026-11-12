// Fixed-point arithmetic shared by the CPU reference and the CUDA kernels.
//
// This reproduces, operation for operation, the software model of the
// fixed-point FFN accelerators used in the CSEE 4868 BERT SoC project
// (bert_precision.h): round-to-nearest conversion, saturation after every
// add and multiply, and the same shift rules when data and weights use
// different fractional widths. Because every step matches, a GPU kernel built
// on these helpers produces bit-identical results to the CPU model.
//
// Two quirks of the original model are kept on purpose so results match it:
//   * saturate() only clips when |x| >= 2^WL (not 2^(WL-1)).
//   * negative overflow saturates to +2^(WL-1) instead of -2^(WL-1).
// See README "Findings".
#pragma once

#include <math.h>

#if defined(__CUDACC__)
#define BERT_HD __host__ __device__ __forceinline__
#else
#define BERT_HD inline
#endif

namespace bert {

struct FixedFormat {
    int wl = 32;  // total word length in bits
    int fl = 16;  // fractional bits
};

struct FixedLayer {
    bool enabled = false;
    FixedFormat data;    // activations, bias, accumulator
    FixedFormat weight;  // weights
};

BERT_HD long long fx_saturate(long long res, int wl) {
    const long long lim = 1LL << wl;
    if (res >= 0) {
        if (res >= lim) res = (1LL << (wl - 1)) - 1;
    } else {
        if (-res >= lim) res = (1LL << (wl - 1));  // original model's behavior
    }
    return res;
}

BERT_HD int fx_from_float(float v, int fl, int wl) {
    double f = (double)v * (double)(1LL << fl);
    long long r = (long long)round(f);
    return (int)fx_saturate(r, wl);
}

BERT_HD float fx_to_float(long long v, int fl) {
    return (float)((double)v / (double)(1LL << fl));
}

BERT_HD int fx_add(int a, int b, int wl) {
    return (int)fx_saturate((long long)a + (long long)b, wl);
}

// a: data (data.fl fractional bits), b: weight (w_fl fractional bits)
BERT_HD int fx_mul(int a, int b, int wl, int fl, int w_fl) {
    long long la = a, lb = b, res;
    if (fl > w_fl) {
        lb = lb << (fl - w_fl);
        res = (la * lb) >> fl;
    } else if (fl < w_fl) {
        la = la << (w_fl - fl);
        res = (la * lb) >> (w_fl + (w_fl - fl));
    } else {
        res = (la * lb) >> fl;
    }
    return (int)fx_saturate(res, wl);
}

// One output of a fixed-point fully connected layer, accumulated in the same
// order as the reference (k = 0..K-1, then bias).
BERT_HD float fx_dot_bias(const float* a_row, const float* w, int ldw, int col,
                          float bias, int K, const FixedLayer& c) {
    int acc = 0;
    for (int k = 0; k < K; ++k) {
        int wf = fx_from_float(w[(long long)k * ldw + col], c.weight.fl, c.weight.wl);
        int xf = fx_from_float(a_row[k], c.data.fl, c.data.wl);
        acc = fx_add(acc, fx_mul(xf, wf, c.data.wl, c.data.fl, c.weight.fl), c.data.wl);
    }
    acc = fx_add(acc, fx_from_float(bias, c.data.fl, c.data.wl), c.data.wl);
    return fx_to_float(acc, c.data.fl);
}

}  // namespace bert
