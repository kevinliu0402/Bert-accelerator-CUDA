# Test plan: from compiler output to GPU execution (Tesla T4)

**Question:** for my own GEMM kernels, can I predict from the generated SASS alone (instruction mix, registers, shared-memory layout) what the hardware counters will show, and does that explain the speed differences?

Written **before** running anything. Every prediction gets checked against measurement in `results/perf/SUMMARY.md`.

## Device under test
| | Tesla T4 (TU104, sm_75) |
|---|---|
| SMs | 40, each 4 sub-partitions (SMSP) × 16 FP32 lanes = 64 FP32/SM |
| Peak FP32 | 40 × 64 × 2 × 1.59 GHz = **8.1 TFLOP/s** (boost; 70 W card, clocks drop under load) |
| Per SM | 64K registers, max 32 warps / 16 blocks, ≤64 KB shared memory, 32 banks × 4 B |
| Memory | 4 MB L2, GDDR6 **320 GB/s** → ridge point ≈ 25 FLOP/B |

Register allocation: per warp, rounded up to 256 registers (8 per thread). Blocks/SM from registers = ⌊65536 / (256 threads × R_rounded)⌋.

## Workload
FFN1 GEMM, C[M×N] = A[M×K]·W[K×N] + bias. N = 512, K = 128.
- **B=64:** M = 2048 (the 97%-of-cuBLAS case). 268 MFLOP. Compulsory traffic 5.5 MB → intensity ≈ 49 FLOP/B → *should be compute-bound*.
- **B=1:** M = 32 (the latency case).

Kernels: `naive`, `tiled`, `reg32` (2×2 per thread), `reg64` (4×4 per thread, PAD=1), `reg64_pad0`, `reg64_pad4`, `cublas`.

## Method
1. **Static (compiler output):** `nvcc --resource-usage` (registers, spills, shared memory); `cuobjdump -sass` → per-kernel opcode counts and the main loop body (`sass_stats.py`).
2. **Runtime API:** `cudaFuncGetAttributes` + `cudaOccupancyMaxActiveBlocksPerMultiprocessor` → registers, local memory (spills), blocks/SM, waves.
3. **Timing:** CUDA events, 200 iterations after warm-up, median of 5 reps; correctness checked against a CPU GEMM.
4. **Counters:** Nsight Compute, one launch per kernel, `--clock-control none` so clocks match the timing runs.
5. **Register sweep:** rebuild with `-maxrregcount` = 32, 40, 48, 56, 64, 80, 96, 128 and rerun reg64.

## Predictions

### P1. Inner-loop instruction mix decides the bottleneck (the core model)
Per SM, one `kk` step of the inner loop costs:
- FMA pipe: each warp FFMA takes 2 cycles on a 16-lane SMSP, and 4 SMSPs run in parallel → `FFMA × 2 / 4` cycles per warp.
- Shared memory: 1 wavefront per cycle per SM (32 banks × 4 B) → `LDS wavefronts` cycles per warp.

| Kernel | FFMA / kk / thread | LDS / kk (predicted SASS) | FMA cycles : LDS cycles | Predicted limiter |
|---|---|---|---|---|
| tiled | 1 | 1 Ws + ¼ As (As row → `LDS.128`) | 0.5 : 1.25 | shared memory, FMA ≤ 40% |
| reg32 | 4 | 2 + 2 (pad 33 breaks alignment → scalar) | 2 : 4 | shared memory, FMA ≤ 50% |
| reg64 (pad 1) | 16 | 4 + 4 scalar | 8 : 8 | balanced |
| reg64_pad0 | 16 | 1 `LDS.128` + 4 | 8 : 5 | FMA, **but** 16-way conflicts on the A-tile stores |
| reg64_pad4 | 16 | 1 `LDS.128` + 4 | 8 : 5 | FMA, only 2-way store conflicts |
| naive | 1 | 0 LDS, 2 LDG | — | L1/L2 load path |

**P1a:** SASS shows `LDS.128` for the A operand only in pad0 and pad4, and scalar `LDS` in pad1 (row stride 65 floats isn't 16-byte aligned).
**P1b:** Shared-memory wavefronts per warp per k-tile (16 kk + 4 stores): pad1 = 128 + 4 = **132**, pad0 = 80 + 64 = **144**, pad4 = 80 + 8 = **88**. So **pad4 fastest; pad0 ≈ pad1** (the store conflicts eat the saved loads).
**P1c:** FMA-pipe utilization ranks reg64_pad4 > reg64 > reg32 > tiled > naive.

### P2. Static × trip count = dynamic count (does my SASS reading predict the counter exactly?)
- FFMA thread-instructions = M·N·K = **134,217,728** for every kernel at B=64 (no wasted lanes; tiles divide evenly).
- reg64 shared-load warp-instructions = 256 blocks × 8 warps × 8 k-tiles × 16 kk × 8 = **2,097,152**; pad4/pad0: × 5/8 = **1,310,720**.
- Bank conflicts: pad1 ≈ 0; pad0 ≈ 256 × 8 × 8 × 4 × 15 = **~983k** extra store wavefronts; pad4 ≈ **~66k**.

### P3. Registers → occupancy → (not much) speed
- reg64 uses **64–96 registers**: 4×4 accumulators + 4 + 4 operands + addresses. That gives 4 / 3 / 2 blocks per SM → 100 / 75 / 50% theoretical occupancy.
- `-maxrregcount` sweep: below ~48 the compiler **spills** (`STL`/`LDL` appear, local memory > 0) and time jumps sharply. Above the no-spill point, more occupancy buys **< 10%**, because each warp already has 16 independent FFMAs per kk (ILP hides latency). *"Occupancy isn't the goal; hiding latency is."*
- **Wave quantization:** 256 blocks at 4/SM = 1.6 waves; at 3/SM = 2.13 → 3 waves. A step in time where blocks/SM changes would be wave quantization, not the per-warp speed.

### P4. Memory isn't the limit at B=64
- DRAM read ≈ 1.3 MB (A + W once, re-reads hit the 4 MB L2), write ≈ 4 MB. At ~110 µs that is ~50 GB/s, **~15% of 320 GB/s**.
- Achieved FP32 is ~30% of peak (2,438 of 8,100), so SOL shows **neither** compute nor memory saturated → the answer is in the stall reasons and the clock.
- Under load the SM clock is **below the 1.59 GHz boost** (70 W card). Measured by ncu `sm__cycles_elapsed.avg.per_second`.

### P5. Stall reasons match the limiter
- naive: **long scoreboard** (waiting on global loads).
- tiled: **barrier** (32-warp blocks) + **MIO throttle / short scoreboard** (shared memory).
- reg64: **not selected / math pipe throttle** grows, short scoreboard shrinks vs reg32.
- B=1 (M=32): only 8–64 blocks on 40 SMs → SOL < 10%, stall = long scoreboard + barrier. It is a **latency** problem; occupancy and SASS tweaks barely matter (matches the original CUDA-graph finding).

### P6. cuBLAS
Kernel name `*sgemm*` (no tensor cores; FP32 math mode). Predict **more registers (≥128) and lower occupancy (≤25%)** than mine, `LDS.128` on both operands, larger tile per thread. It wins with ILP, not occupancy.

## Pass/fail
- Correctness: every kernel max relative error < 1e-3 vs CPU, or the run is invalid.
- Timing: CV across 5 reps < 5%, or rerun.
- A prediction **passes** if the measured value is within the stated range or exactly equal for counts (P2). Every miss gets an explanation in the walkthrough, not a silent edit of the prediction.
- If ncu is blocked (`ERR_NVGPUCTRPERM`), P1b/P2/P4/P5 are reported as **not measured**; static + timing parts still stand.
