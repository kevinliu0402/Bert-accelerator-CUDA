# bert-cuda

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
- **Saturation quirks:** `saturate()` only clips when |x| ≥ 2^WL, not 2^(WL−1). Negative overflow also saturates to **+**2^(WL−1), flipping the sign. `fixed_point.h` keeps both behaviors so results match the original model bit for bit.

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

*Fill this in from `results/` after running on a GPU.*
