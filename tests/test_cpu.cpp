// CPU-only check (no GPU needed): runs the reference model on the 10 project
// sentences in floating point and with the SoC's fixed-point FFN formats.
//   make cpu-test DATA=path/to/data
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "cpu_ref.h"

using namespace bert;

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "data";
    Weights w;
    std::string err;
    if (!load_weights(dir, &w, &err)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 2;
    }
    const FixedConfig none;
    const FixedConfig soc = soc_fixed_config();
    int ok_float = 0, ok_fixed = 0;
    std::printf("%-11s %5s %9s %9s %5s | %9s %9s %5s\n", "sentence", "label", "neg", "pos", "pred", "neg(fx)",
                "pos(fx)", "pred");
    for (int i = 0; i < kNumSentences; ++i) {
        std::vector<float> x;
        if (!load_sentence(dir, i, &x, &err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 2;
        }
        float lf[2], lx[2];
        forward_cpu(w, x.data(), kMasks[i], none, lf);
        forward_cpu(w, x.data(), kMasks[i], soc, lx);
        int pf = lf[1] > lf[0], px = lx[1] > lx[0];
        ok_float += pf == kLabels[i];
        ok_fixed += px == kLabels[i];
        std::printf("%-11s %5s %9.5f %9.5f %5s | %9.5f %9.5f %5s\n", kSentences[i], kLabels[i] ? "pos" : "neg", lf[0],
                    lf[1], pf ? "pos" : "neg", lx[0], lx[1], px ? "pos" : "neg");
    }
    std::printf("\nfloat accuracy: %d/10   fixed-point FFN (8-bit / 15-bit) accuracy: %d/10\n", ok_float, ok_fixed);
    return ok_float == kNumSentences ? 0 : 1;
}
