# Bert-accelerator-CUDA

This project runs the single-block BERT sentiment classifier from the CSEE 4868 System-on-Chip Platforms project (Columbia, Fall 2025) on an NVIDIA GPU with hand-written CUDA kernels. In the course project, the two feed-forward (FFN) layers ran on Catapult-HLS accelerators inside an ESP RISC-V SoC on a Xilinx VC707 FPGA. Here the whole model runs on the GPU, including those FFN layers in the project's own fixed-point format.

## Model

- 32 tokens and a hidden size of 128.
- 2 attention heads.
- An FFN with 128 → 512 → 128 layers and GELU.
- A pooler and a 2-way classifier.
- 10 test sentences: 5 positive and 5 negative.

## What's here

| File | Contents |
|---|---|
| `src/kernels.cu` | Four GEMM kernels, each with bias and GELU/tanh fused into the epilogue. Plus a cuBLAS path, a fixed-point GEMM, masked attention, and small helper kernels. |
| `src/bert_gpu.cu` | The full forward pass for any batch of sentences. It can launch kernels one by one or replay them as a single CUDA graph. |
| `src/cpu_ref.cpp` | A CPU reference model and loaders for the project's `.bin` weights. |
| `src/fixed_point.h` | The project's fixed-point arithmetic, shared by the CPU and GPU code. |
| `tests/test_kernels.cu` | Unit tests for the CUDA kernels against the CPU reference. |
| `tests/test_cpu.cpp` | A CPU-only check that runs without a GPU. |
| `apps/bert_infer.cu` | Accuracy, logit error, and latency for each kernel, and the fixed-point precision sweep. |
| `bench/bench_gemm.cu` | GEMM throughput from 1 to 256 sentences, compared with cuBLAS. |
| `scripts/profile.sh` | Nsight Compute and Nsight Systems profiling. |

### GEMM kernels

All of them compute C = act(A·W + b), with A as tokens × in and W as in × out.

| Kernel | Idea |
|---|---|
| `naive` | One thread per output, reading both operands from global memory. |
| `tiled` | 32×32 shared-memory tiles; the W tile is padded to avoid bank conflicts. |
| `reg32` | 32×32 block tile with 2×2 outputs per thread held in registers. It launches more blocks, which suits 32-token inputs. |
| `reg64` | 64×64 block tile with 4×4 outputs per thread. It reuses more data per load, which suits large batches. |
| `cublas` | `cublasSgemm` followed by a fused bias/activation kernel; used as the reference point. |

A single sentence gives FFN shapes of 32×512×128 and 32×128×512. They are too small to fill the GPU: `reg64` launches only 8 blocks for FFN1. The benchmark therefore reports block counts next to SM counts, and compares one sentence, batches of sentences, and CUDA graphs. The graphs exist to remove launch overhead, which dominates at this size.

## Correctness

- **FP32 kernels:** every kernel is checked against the CPU on shapes that are and are not multiples of the tile sizes, with error bounded by Σ|a·w|.
- **Fixed-point GEMM:** the GPU kernel must be bit-exact with the CPU model.
- **Full model:** the run checks accuracy on the 10 sentences and the largest logit difference from the CPU.

## Findings in the course's reference model

- **Uninitialized accumulators:** `BertMatrixVector_float` adds into output buffers that came from `malloc()` and were never cleared. On Linux, fresh pages happen to be zero. On macOS, the small pooler and classifier buffers start with garbage, and **every sentence is classified "Positive" (5/10)**. Zeroing the buffers restores 10/10. `cpu_ref.cpp` clears its accumulators.
- **Saturation quirks:** `saturate()` only clips when |x| ≥ 2^WL, not 2^(WL−1). Negative overflow also saturates to a positive +2^(WL−1) instead of −2^(WL−1), flipping the sign. `fixed_point.h` keeps both behaviors so results match the original model bit for bit.

## Fixed-point precision vs. accuracy

These results come from the CPU model (`--sweep` runs the same experiment on the GPU). Each cell is accuracy on the 10 sentences. Data and weights use the same format in both FFN layers, so FL = WL − IL.

| WL \ IL | 1 | 2 | 3 | 4 | 6 | 8 | 16 |
|---|---|---|---|---|---|---|---|
| 8–12 | 5 | 5 | 5 | 5 | 5 | 5 | |
| 14 | 5 | 8 | 6 | 5 | 5 | 5 | |
| 16 | 5 | 9 | 9 | 8 | 5 | 5 | |
| 18 | 5 | **10** | 9 | 9 | 8 | 5 | 5 |
| 20 | 5 | **10** | **10** | **10** | 9 | 8 | 5 |
| 24 | 5 | **10** | **10** | **10** | **10** | **10** | 5 |
| 32 | 5 | 5 | **10** | **10** | **10** | **10** | **10** |

- **Smallest format that keeps 10/10:** 18 bits with 2 integer bits. Below 16 bits, the model collapses to predicting one class (5/10).
- **One integer bit always fails,** because the FFN activations exceed ±2.
- **Large fractional parts also fail,** because the original `mul_fixed` overflows its 64-bit intermediate product (WL 32 with 1–2 integer bits).
- **This differs from the FPGA hardware:** the SoC accelerators use Catapult `ac_fixed` types, which round and saturate differently from this software model. These numbers describe the course's software model, not the FPGA results.

## Build and run

You need an NVIDIA GPU with the CUDA toolkit (11.4 or newer), or Google Colab. The model files go in `data/`; they come from the course project and are not in this repo.

```bash
make ARCH=sm_75          # T4; use sm_86 for RTX 30xx, sm_80 for A100
make test                # kernel unit tests
make run DATA=data       # BERT on GPU: FP32 and fixed point
make sweep DATA=data     # precision sweep
make bench               # GEMM benchmark, writes results/bench_gemm.csv
make cpu-test DATA=data  # no GPU needed
```

On Colab, run `scripts/package_for_colab.sh`, then open `notebooks/colab.ipynb` and upload the zip.

## Results

These results come from an NVIDIA Tesla T4 on Google Colab (40 SMs, CUDA 12). The raw CSVs are in `docs/results-t4/`.

### End-to-end inference (10 sentences, FP32 FFN)

Every kernel classifies **10/10** sentences. The largest logit difference from the CPU reference is **1.8×10⁻⁶**.

| GEMM | 1 sentence, eager | 1 sentence, CUDA graph | FFN layers only | 10-sentence batch | Sentences/s |
|---|---|---|---|---|---|
| naive | 123.5 µs | 111.5 µs | 43.9 µs | 424.1 µs | 23,600 |
| tiled | 128.8 µs | 120.0 µs | 48.3 µs | 212.4 µs | 47,100 |
| reg32 | 123.9 µs | 117.0 µs | 47.4 µs | 178.7 µs | 56,000 |
| reg64 | 187.5 µs | 183.3 µs | 77.0 µs | 230.1 µs | 43,500 |
| cuBLAS | 128.3 µs | 63.2 µs | 32.3 µs | 125.0 µs | 80,000 |

With the FFN in 18-bit fixed point (`--fixed 18,16`), accuracy stays at **10/10** and the largest logit error is 8.9×10⁻⁵. The fixed-point FFN takes 393 µs, about 8× the FP32 version. Its kernel uses one thread per output and does every conversion in FP64, which runs at 1/32 of the FP32 rate on a T4.

### GEMM throughput (`make bench`)

| Shape | naive | tiled | reg32 | reg64 | cuBLAS |
|---|---|---|---|---|---|
| FFN1, 1 sentence (32×512×128) | 240 | 202 | 207 | 160 | 323 GFLOP/s |
| FFN1, 64 sentences (2048×512×128) | 523 | 647 | 1,420 | **2,438 (97%)** | 2,505 |
| FFN1, 256 sentences (8192×512×128) | 648 | 915 | 1,343 | **2,356 (87%)** | 2,706 |
| FFN2, 1 sentence (32×128×512) | 183 | 79 | 81 | 57 | 307 |
| FFN2, 256 sentences (8192×128×512) | 608 | 904 | 1,341 | **2,128 (71%)** | 2,978 |

A later run on CUDA 13.0 (see [Compiler output vs. hardware counters](#compiler-output-vs-hardware-counters-perf)) measured `reg64` at 86–87% of cuBLAS for 64 sentences and 86% for 256. The 97% point did not reproduce, so treat 86–97% as the range across toolkits and runs.

### What the numbers show

- **Register blocking pays off once the GPU is full.**
  - At 64 sentences, `reg64` is 4.7× faster than `naive` and reaches 97% of cuBLAS on FFN1.
  - At 10 sentences, `reg32` runs the whole model 2.4× faster than `naive`.
- **One sentence is too small to fill the GPU.**
  - FFN2 for one sentence (N = 128) gives `tiled`/`reg32` only 4 blocks and `reg64` only 2, on 40 SMs. Those kernels reach just 19–26% of cuBLAS.
  - `naive` uses smaller blocks, so it gets 16 of them and is the fastest custom kernel at this size.
- **The Nsight Systems trace of a one-sentence pass (`reg32`) confirms this.**
  - The seven GEMMs with 128 output columns (Q, K, V, dense, FFN2, pooler) each launch 4 blocks and average 14.9 µs.
  - FFN1 launches 16 blocks and takes 10.0 µs.
  - Attention takes 8.4 µs, and the 2-output classifier, which launches 1 block, takes 8.4 µs.
- **CUDA graphs help cuBLAS far more than the custom kernels.**
  - Graph replay halves cuBLAS's single-sentence latency (128 → 63 µs), because each cuBLAS call has host-side overhead.
  - For the custom kernels it saves only 6–10%. Their time is spent on the GPU, not on launches.

### Next optimizations

- **Split-K or smaller tiles** for the N = 128, K = 512 layers at one sentence, so they launch more blocks than there are SMs.
- **An integer-only fixed-point kernel** that quantizes the weights once, removing the per-element FP64 conversions.

## Compiler output vs. hardware counters (`perf/`)

A second study asks whether the generated SASS predicts what the hardware counters show. It ran on a Tesla T4 on Colab with CUDA 13.0 and Nsight Compute 2025.3. The predictions were written first, in [perf/TEST_PLAN.md](perf/TEST_PLAN.md). The full analysis, including the misses, is in [perf/WALKTHROUGH.md](perf/WALKTHROUGH.md), and the generated tables are in `results/perf/SUMMARY.md`.

- **Static SASS × trip count = dynamic count.**
  - FFMA and shared-load counts match Nsight Compute exactly for all 8 kernel variants (for example, 1,507,328 `LDS` for `reg64`).
  - The static `STS` count overestimates, because the compiler emitted a second load path that never runs.
- **The compiler beat my model.**
  - With the A tile padded to 65 floats, I predicted scalar shared loads.
  - ptxas instead issued `LDS.128` loads that start 4 bytes early and read the padding word. That gives 92 loads per k-tile instead of 128.
- **Padding is a bank-conflict vs. vector-load trade-off.**

  | A-tile padding | 0 | 1 | 4 |
  |---|---|---|---|
  | Measured wavefronts per warp per k-tile (my model gives the same numbers) | 164 | 116 | 108 |
  | Time (µs) | 138 | 119 | 114 |

  Padding 4 takes `reg64` from 86% to 90% of cuBLAS.
- **Occupancy wasn't the limiter.**
  - Going from 75% to 100% theoretical occupancy (via `__launch_bounds__` or `-maxrregcount`) changed time by ≤ 2%, because the load/store pipe was the bottleneck (67–87% busy, top stall `mio_throttle`).
  - Spilling at 32 registers cost 19–23%.
- **cuBLAS (`volta_sgemm_128x64_nn`) does more work per thread.**
  - It keeps the FMA pipe 80% busy at 33% achieved occupancy, against 48% for my best kernel.
  - It computes 64 outputs per thread to my 16. That's the next optimization.

```bash
ARCH=sm_75 ./perf/run_perf.sh        # or open perf/colab_perf.ipynb on Colab
STAGES=3 ./perf/run_perf.sh          # rerun only the register sweep
```
