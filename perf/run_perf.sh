#!/usr/bin/env bash
# Compiler output -> GPU execution, on one GPU (made for the Colab T4).
#   ARCH=sm_75 ./perf/run_perf.sh          (from the repo root)
# Writes everything to results/perf/. Each stage logs what it ran.
set -uo pipefail
cd "$(dirname "$0")/.."
ARCH=${ARCH:-sm_75}
OUT=results/perf
mkdir -p "$OUT" bin
NVFLAGS="-O3 -std=c++17 -arch=$ARCH -lineinfo -Isrc"
LIBS=-lcublas
SIZES_MAIN="2048 32 8192"   # FFN1 with 64, 1 and 256 sentences (N=512, K=128)
STAGES=${STAGES:-1234}      # e.g. STAGES=3 reruns only the register sweep
run() { [[ "$STAGES" == *"$1"* ]]; }

say() { echo; echo "=== $*"; }

say "0. system"
{
    nvidia-smi --query-gpu=name,compute_cap,driver_version,clocks.max.sm,power.limit --format=csv
    nvcc --version | tail -2
    command -v ncu >/dev/null && ncu --version | tail -1
} | tee "$OUT/system.txt"

say "1. default build: registers, spills, SASS"
nvcc $NVFLAGS --resource-usage perf/perf_gemm.cu -o bin/perf_gemm $LIBS > "$OUT/resource_usage_default.txt" 2>&1 \
    || { cat "$OUT/resource_usage_default.txt"; exit 1; }
if run 1; then
    rm -f "$OUT/sass_summary.csv"
    cuobjdump -sass bin/perf_gemm > "$OUT/sass_default.txt"
    python3 perf/sass_stats.py "$OUT/sass_default.txt" --tag default --out "$OUT"
fi

if run 2; then
    say "2. timing: every kernel, 3 sizes, plus a repeat of the main size"
    rm -f "$OUT/timing.csv"
    for M in $SIZES_MAIN; do
        ./bin/perf_gemm --kernel all --M "$M" --tag default --csv "$OUT/timing.csv"
    done
    ./bin/perf_gemm --kernel all --M 2048 --tag rerun --csv "$OUT/timing.csv"
fi

if run 3; then
    say "3. register sweep: -maxrregcount with __launch_bounds__ removed"
    rm -f "$OUT/regsweep.csv"
    # Drop old sweep rows from the SASS summary so a rerun of stage 3 alone doesn't duplicate them.
    [ -f "$OUT/sass_summary.csv" ] && python3 -c "import csv,sys; r=list(csv.reader(open(sys.argv[1]))); csv.writer(open(sys.argv[1],'w',newline='')).writerows([x for x in r if not (x[0].startswith('r') and x[0][1:].isdigit())])" "$OUT/sass_summary.csv"
    for R in 32 40 48 56 64 80 96 128; do
        nvcc $NVFLAGS -DBERT_NO_LAUNCH_BOUNDS -maxrregcount=$R --resource-usage perf/perf_gemm.cu \
            -o "bin/perf_gemm_r$R" $LIBS > "$OUT/resource_usage_r$R.txt" 2>&1 || { cat "$OUT/resource_usage_r$R.txt"; exit 1; }
        cuobjdump -sass "bin/perf_gemm_r$R" > "$OUT/sass_r$R.txt"
        python3 perf/sass_stats.py "$OUT/sass_r$R.txt" --tag "r$R" --out "$OUT" | grep -E 'reg64 |reg64_pad4'
        for M in 2048 8192; do
            for k in reg64 reg64_pad4; do
                ./bin/perf_gemm_r$R --kernel $k --M $M --tag "r$R" --csv "$OUT/regsweep.csv" | tail -1
            done
        done
    done
    # Does -maxrregcount still apply when __launch_bounds__ is present? Test it instead of trusting a doc.
    nvcc $NVFLAGS -maxrregcount=32 --resource-usage perf/perf_gemm.cu -o bin/perf_gemm_lb_r32 $LIBS \
        > "$OUT/resource_usage_lb_r32.txt" 2>&1
    ./bin/perf_gemm_lb_r32 --kernel reg64 --M 2048 --tag lb_r32 --csv "$OUT/regsweep.csv" | tail -1
fi

NCU_OK=$(cat "$OUT/ncu_ok.txt" 2>/dev/null || echo 0)
if run 4; then
    say "4. Nsight Compute counters (one launch per kernel, clocks not locked)"
    NCU_OK=0
    if command -v ncu >/dev/null; then
        ncu --kernel-name regex:gemm --launch-count 1 ./bin/perf_gemm --kernel naive --M 32 --iters 1 --reps 1 \
            --warmup 0 --warm-ms 0 > "$OUT/ncu_probe.txt" 2>&1
        if grep -q "ERR_NVGPUCTRPERM\|==ERROR==" "$OUT/ncu_probe.txt"; then
            echo "ncu cannot read counters here:"; grep "ERR\|ERROR" "$OUT/ncu_probe.txt" | head -3
        else
            NCU_OK=1
        fi
    else
        echo "ncu not found"
    fi
    if [ "$NCU_OK" = 1 ]; then
        METRICS=$(python3 perf/ncu_metrics.py)
        EXTRA=()
        [ -n "$METRICS" ] && EXTRA=(--metrics "$METRICS")
        mkdir -p "$OUT/ncu"
        run_ncu() {  # kernel M
            local name="$1_M$2"
            ncu --clock-control none --set full --import-source yes "${EXTRA[@]}" \
                --kernel-name regex:gemm --launch-skip 2 --launch-count 1 -f -o "$OUT/ncu/$name" \
                ./bin/perf_gemm --kernel "$1" --M "$2" --iters 1 --reps 1 --warmup 2 --warm-ms 0 \
                > "$OUT/ncu/$name.log" 2>&1
            ncu --import "$OUT/ncu/$name.ncu-rep" --csv --page raw > "$OUT/ncu/$name.csv" 2>/dev/null
            echo "  $name: $(grep -c '' "$OUT/ncu/$name.csv") csv lines"
        }
        for k in naive tiled reg32 reg64 reg64_pad0 reg64_pad4 reg64_minb2 reg64_minb3 reg64_minb4 cublas; do
            run_ncu $k 2048
        done
        for k in naive reg32 reg64 cublas; do run_ncu $k 32; done
    fi
    echo "$NCU_OK" > "$OUT/ncu_ok.txt"
fi

say "5. analysis"
python3 perf/analyze.py --out "$OUT" || echo "analyze.py failed (results are still saved)"
echo; echo "done: $OUT"
