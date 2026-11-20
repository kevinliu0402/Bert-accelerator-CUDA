#!/usr/bin/env python3
"""Shared-memory wavefront model for the GEMM kernels, checked against Nsight Compute.

Shared memory serves one "wavefront" (one pass over its 32 banks) per cycle. The
model counts wavefronts per warp per k-tile:

    wavefronts = LDS instructions                   (from the SASS main loop)
               + 1 extra per LDS.128                (measured cost on this GPU)
               + STS instructions that execute      (from the tile geometry)
               + extra passes from store bank conflicts (from the bank formula)

Measured value = (l1tex shared ld + st wavefronts) / (blocks x warps x k-tiles).

    python3 perf/wavefront_model.py [results/perf]
"""
import os
import sys
from collections import Counter

import pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/perf"
M, N, K, BANKS = 2048, 512, 128, 32

# Tile geometry of each kernel: block tile BMxBN, K-slice BK, threads per block,
# padding on the A-tile rows, and how each tile is stored (see src/kernels.cu).
KERNELS = {
    "tiled":      dict(BM=32, BN=32, BK=32, NT=1024, PAD=0, layout="tiled"),
    "reg64":      dict(BM=64, BN=64, BK=16, NT=256, PAD=1, layout="regblock"),
    "reg64_pad0": dict(BM=64, BN=64, BK=16, NT=256, PAD=0, layout="regblock"),
    "reg64_pad4": dict(BM=64, BN=64, BK=16, NT=256, PAD=4, layout="regblock"),
}


def store_word_addresses(g, warp, it):
    """Word addresses one warp writes in one store instruction (A tile, then W tile)."""
    a, w = [], []
    for lane in range(32):
        tid = warp * 32 + lane
        if g["layout"] == "tiled":           # As[ty][tx], Ws[ty][tx+pad 1]; block is 32x32 threads
            ty, tx = tid // 32, tid % 32
            a.append(ty * 32 + tx)
            w.append(ty * 33 + tx)
        else:                                # regblock: loops i = tid + it*NT
            i = tid + it * g["NT"]
            r, c = i // g["BK"], i % g["BK"]
            a.append(c * (g["BM"] + g["PAD"]) + r)   # As[c][r]: stored K-major (transposed)
            r2, c2 = i // g["BN"], i % g["BN"]
            w.append(r2 * g["BN"] + c2)              # Ws[r][c]
    return a, w


def passes(addrs):
    """Wavefronts for one warp store: the worst bank's number of distinct words."""
    per_bank = Counter()
    for word in set(addrs):
        per_bank[word % BANKS] += 1
    return max(per_bank.values())


def model(name, g, sass):
    lds = int(sass.loop_lds)
    lds128 = int(sass.loop_lds128)
    if g["layout"] == "tiled":
        iters = 1
    else:
        iters = g["BM"] * g["BK"] // g["NT"]          # A-tile stores per thread per k-tile (= W-tile)
    sts = 2 * iters
    extra = 0
    for it in range(iters):
        a, w = store_word_addresses(g, 0, it)
        extra += (passes(a) - 1) + (passes(w) - 1)
    return dict(lds=lds, lds128_extra=lds128, sts=sts, conflict_extra=extra,
                model=lds + lds128 + sts + extra)


def measured(name, g):
    path = os.path.join(OUT, "ncu", f"{name}_M{M}.csv")
    if not os.path.exists(path):
        return float("nan")
    d = pd.read_csv(path, dtype=str)
    row = d.iloc[1]
    num = lambda c: float(str(row[c]).replace(",", ""))
    total = num("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum") + \
        num("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum")
    blocks = (M // g["BM"]) * (N // g["BN"])
    warps = g["NT"] // 32
    ktiles = K // g["BK"]
    return total / (blocks * warps * ktiles)


def main():
    s = pd.read_csv(os.path.join(OUT, "sass_summary.csv"))
    s = s[s.tag == "default"].set_index("kernel")
    print(f"{'kernel':<12}{'LDS':>5}{'+.128':>7}{'STS':>5}{'+conf':>7}{'model':>7}{'ncu':>8}")
    for name, g in KERNELS.items():
        r = model(name, g, s.loc[name])
        m = measured(name, g)
        print(f"{name:<12}{r['lds']:>5}{r['lds128_extra']:>7}{r['sts']:>5}{r['conflict_extra']:>7}"
              f"{r['model']:>7}{m:>8.1f}")


if __name__ == "__main__":
    main()
