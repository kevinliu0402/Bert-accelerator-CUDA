#!/usr/bin/env bash
# Profiles the GEMM kernels with Nsight Compute on the one-sentence FFN shapes.
# Needs ncu on PATH and permission to read GPU performance counters
# (on Colab and some cloud VMs this is blocked; nsys still works).
set -euo pipefail
DATA=${1:-data}
mkdir -p results

if ! command -v ncu >/dev/null; then
    echo "ncu (Nsight Compute) not found; install the CUDA toolkit or use nsys" >&2
    exit 1
fi

# One FFN pass per GEMM kind; profile the FFN kernels only.
for g in naive tiled reg32 reg64; do
    ncu --target-processes all \
        --kernel-name regex:'gemm_' --launch-count 2 \
        --section SpeedOfLight --section Occupancy --section LaunchStats \
        --section MemoryWorkloadAnalysis --section WarpStateStats \
        --csv --page details \
        ./bin/bert_infer --data "$DATA" --gemm "$g" --iters 1 \
        > "results/ncu_${g}.csv" || true
    echo "wrote results/ncu_${g}.csv"
done

# Timeline of a full forward pass, eager vs. CUDA graph.
if command -v nsys >/dev/null; then
    nsys profile -o results/timeline --force-overwrite true --trace cuda \
        ./bin/bert_infer --data "$DATA" --gemm reg32 --iters 20
    nsys stats --report cuda_gpu_kern_sum results/timeline.nsys-rep > results/nsys_kernels.txt || true
    echo "wrote results/timeline.nsys-rep and results/nsys_kernels.txt"
fi
