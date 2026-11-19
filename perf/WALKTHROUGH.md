# Walkthrough: from compiler output to GPU execution

Parts 1–5 explain the method, with short questions and answers along the way. Part 6 has the measured results from a Tesla T4.

Files: [TEST_PLAN.md](TEST_PLAN.md) (predictions, written before running) · [perf_gemm.cu](perf_gemm.cu) (the harness) · [sass_stats.py](sass_stats.py) (reads the assembly) · [run_perf.sh](run_perf.sh) (the whole experiment) · [analyze.py](analyze.py) (prediction vs measurement) · [colab_perf.ipynb](colab_perf.ipynb).

---

## Part 1. The toolchain: what the compiler actually produces

```
kernels.cu --nvcc front end--> PTX (virtual ISA, infinite registers) --ptxas--> SASS (real ISA for sm_75) --> cubin in the binary
```
- **PTX** is portable and uses virtual registers (`%f1`, `%f2`, …). It says nothing about the real register count.
- **ptxas** does the real work: **register allocation**, **instruction scheduling**, unrolling decisions, and picks real opcodes (`FFMA`, `LDS.U.128`, …). The result is **SASS**.
- `nvcc --resource-usage` prints what ptxas decided: `Used 72 registers, 8256 bytes smem, 0 bytes spill stores`.
- `cuobjdump -sass` prints the SASS. That is what the GPU executes, so it is what we read.

**Q1. Why read SASS and not PTX when you care about performance?**
> PTX is before register allocation and scheduling. Spills, vector-load choices (`LDS.128` vs four `LDS`), and the final instruction count only exist in SASS. Two kernels with similar PTX can have very different SASS.

## Part 2. Reading the main loop of `reg64`

Each thread computes a 4×4 tile of outputs. One k-tile iteration (BK = 16) does:
1. **LDG** global loads of its share of the A and W tiles (4 + 4 floats per thread),
2. **STS** stores into shared memory, `BAR.SYNC`,
3. 16 × `kk` steps, fully unrolled: load `a[4]` and `w[4]` from shared memory (**LDS**), then 16 **FFMA** (the 4×4 outer product),
4. `BAR.SYNC`, branch back.

So per thread per k-tile: **256 FFMA** and, with scalar loads, **128 LDS**. FFMA:LDS = 2.

`sass_stats.py` finds that loop automatically: the backward `BRA` whose body has the most FFMAs. The listing lands in `results/perf/sass_loops/default_reg64.txt`.

**Q2. Why does a bigger per-thread tile (TM×TN) raise the FFMA:LDS ratio?**
> An outer product of a TM-vector and a TN-vector costs TM + TN loads and TM × TN FMAs. 2×2: 4 loads, 4 FMAs (1:1). 4×4: 8 loads, 16 FMAs (2:1). 8×8: 16 loads, 64 FMAs (4:1). That is the whole point of register blocking: the registers become the cache.

**Q3. What's the cost?**
> Registers. 16 accumulators + 8 operands + addresses and loop state. More registers per thread → fewer warps fit on the SM.

## Part 3. Registers → occupancy (do this math by hand)

T4 SM: 65,536 registers, max 32 warps (1,024 threads), max 16 blocks. Registers are allocated per warp in units of 256 (so per thread, round up to a multiple of 8).

For a 256-thread block (8 warps):

| Registers/thread | Registers/block | Blocks/SM by registers | Warps | Theoretical occupancy |
|---|---|---|---|---|
| 64 | 16,384 | 4 | 32 | 100% |
| 72 | 18,432 | 3 | 24 | 75% |
| 96 | 24,576 | 2 | 16 | 50% |
| 128 | 32,768 | 2 | 16 | 50% |
| 168 | 43,008 | 1 | 8 | 25% |

**Q4. reg64 at FFN1, 64 sentences, launches 256 blocks. With 4 blocks/SM, how many waves? With 3?**
> 40 SMs × 4 = 160 blocks per wave → 256/160 = 1.6 → **2 waves**, the second 60% full. At 3/SM: 120 per wave → 2.13 → **3 waves**, the last one almost empty. That's **wave quantization**: time moves in steps of whole waves, so a register change that drops one block/SM can cost ~50% here, and nothing at all at another size.

**Q5. Is higher occupancy always faster?**
> No. Occupancy only helps if you need more warps to hide latency. reg64 has 16 independent FFMAs per kk per thread, so each warp has a lot of ILP already. Above the point where latency is hidden, more occupancy buys nothing, and forcing it (fewer registers) can cause **spills** to local memory (`STL`/`LDL`, which go through L1/L2), which is much worse. The register sweep measures exactly this.

### The two knobs, and a trap
- `-maxrregcount=N` (whole file) and `__launch_bounds__(threads, minBlocks)` (per kernel) both cap registers.
- The kernel already has `__launch_bounds__(256)`. Does `-maxrregcount` still apply then? The harness **tests it** (`lb_r32` row) instead of trusting memory, and the sweep builds with `-DBERT_NO_LAUNCH_BOUNDS` so the flag is unambiguous.
- `reg64_minb2/3/4` use `__launch_bounds__(256, MINB)`: "guarantee at least MINB blocks per SM", which caps registers at 65536 / (MINB × 256) = 128 / 85 / 64.

## Part 4. Shared memory banks: why PAD matters

Shared memory has 32 banks, 4 bytes wide. Bank = (word address) mod 32. One warp request is served in one **wavefront** if every thread hits a different bank (or the same word, which is a broadcast). Each extra distinct word in the same bank costs one more wavefront: a **bank conflict**.

The A tile is stored transposed: `As[c][r]` with row length `BM + PAD` = 64 + PAD.

**Stores** (once per k-tile, 4 per thread): in one warp, 16 consecutive threads write `c = 0..15` for the same `r`. Word address = c·(64 + PAD) + r.
- PAD = 0: bank = (64c + r) mod 32 = r → all 16 in **one bank** → 16-way conflict.
- PAD = 1: bank = (65c + r) mod 32 = (c + r) mod 32 → all different → conflict-free.
- PAD = 4: bank = (68c + r) mod 32 = (4c + r) mod 32 → c and c+8 collide → 2-way.

**Loads** (16 times per k-tile): a thread reads `As[kk][ty*4 .. ty*4+3]`, four consecutive floats. The compiler can use one 128-bit `LDS.128` only if the address is 16-byte aligned:
- PAD = 0: row = 64 floats = 256 B ✓. PAD = 4: 68 floats = 272 B = 17 × 16 ✓.
- PAD = 1: 65 floats = 260 B ✗ → four scalar `LDS`.

**Q6. So which PAD wins? Count wavefronts per warp per k-tile.**
> Loads: PAD 1 → 16 × 8 = 128; PAD 0/4 → 16 × (1 + 4) = 80. Stores: PAD 1 → 4; PAD 0 → 4 × 16 = 64; PAD 4 → 4 × 2 = 8. Totals **132 / 144 / 88**. Prediction: PAD 4 best, PAD 0 no better than PAD 1, because the store conflicts eat what the vector loads saved. *This is a compiler × microarchitecture interaction: the layout decides the instruction the compiler can emit, and the same layout decides the bank behavior.*

## Part 5. Which counters, and what they mean

| Question | Nsight Compute metric | Reads as |
|---|---|---|
| How many FMAs ran? | `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum` | thread-level count; must equal M·N·K |
| How many shared loads/stores (warp instr)? | `smsp__inst_executed_op_shared_ld/st.sum` | compare to static SASS × trips × warps |
| Bank conflicts | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld/st.sum` | extra wavefronts |
| Is the FMA unit busy? | `sm__pipe_fma_cycles_active…pct` | execution-unit utilization |
| Is the load/store unit busy? | `sm__inst_executed_pipe_lsu…pct` | |
| Achieved occupancy | `sm__warps_active…pct` | vs theoretical from the register math |
| Real clock | `sm__cycles_elapsed.avg.per_second` | the T4 is a 70 W card; boost isn't guaranteed |
| Speed of light | `sm__throughput`, `gpu__compute_memory_throughput` | % of peak, compute vs memory side |
| Why warps wait | `smsp__average_warps_issue_stalled_<reason>…` | long_scoreboard = global/L1 data; short_scoreboard / mio_throttle = shared memory; barrier = `__syncthreads`; math_pipe_throttle = FMA busy; not_selected = plenty of ready warps |

Two method choices:
- `--clock-control none`: by default ncu **locks clocks to base** (585 MHz on a T4) for repeatability. That would make ncu time incomparable with the event timing, so I unlocked it and record the actual clock instead.
- **Static × trip count = dynamic count** (TEST_PLAN P2). If my reading of the SASS is right, the predicted count of LDS/STS/FFMA equals what the hardware counted, exactly. When it doesn't, either my loop model is wrong (extra code outside the loop, a remainder loop) or the compiler did something I didn't see. Either way I learn something.

**Q7. A kernel is at 30% of peak FLOPs and 15% of DRAM bandwidth. What's the bottleneck?**
> Neither roofline is the limit, so it's **latency or an internal pipe**. Next: FMA-pipe vs LSU-pipe utilization, stall reasons, achieved vs theoretical occupancy, actual clock. For reg64 I expect shared memory issue (LSU / MIO) and barrier stalls, and a clock below boost.

**Q8. Why does the one-sentence case (M = 32) not care about any of this?**
> FFN1 at M = 32 is 8 blocks (reg64) on 40 SMs. 32 SMs idle, ~1 wave of 0.05. It is a **latency** problem: fill the GPU (batching), fuse kernels, or remove launch overhead (CUDA graphs, which halved latency in the original project). Register and bank tweaks move the needle by microseconds at most.

---

## Part 6. Results: Tesla T4, CUDA 13.0, Colab (2026-11-19)

Everything below is from `results/perf/` (generated tables in [SUMMARY.md](../results/perf/SUMMARY.md)). ncu = Nsight Compute 2025.3, one profiled launch per kernel, clocks unlocked.

### Headline (FFN1, M=2048, event timing, median of 5 × 200 launches)
| kernel | regs | blocks/SM (occ.) | µs | GFLOP/s | % of cuBLAS | FMA pipe % (ncu) | LSU pipe % (ncu) |
|---|---|---|---|---|---|---|---|
| naive | 50 | 4 (100%) | 480 | 559 | 21% | 13.5 | 82.8 |
| tiled | 45 | 1 (100%) | 289 | 930 | 36% | 14.8 | 73.5 |
| reg32 | 64 | 4 (100%) | 198 | 1,355 | 52% | 28.3 | 87.3 |
| reg64 | 72 | 3 (75%) | 119 | 2,258 | 86% | 45.1 | 70.4 |
| reg64_pad0 | 69 | 3 (75%) | 138 | 1,946 | 74% | 35.9 | 50.9 |
| **reg64_pad4** | 69 | 3 (75%) | **114** | **2,365** | **90%** | 48.1 | 67.2 |
| cuBLAS `volta_sgemm_128x64_nn` | 122 | 128-thread blocks, 50% theor., 32.8% achieved | 102 | 2,620 | 100% | **80.3** | 42.7 |

Full rerun: reg64 123 µs (86%), pad4 118 µs (89%), cuBLAS 105 µs. All kernels max relative error ≤ 9.4e-7.

⚠️ **The older README's "reg64 = 97% of cuBLAS" (CUDA 12 run) did not reproduce on CUDA 13:** reg64 86–87%, pad4 89–90%. Both runs are real; the toolkit (ptxas and cuBLAS) changed between them, and these data can't separate the two causes.

### Grading the predictions

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| P1a | pad1: 4 scalar `LDS` per kk for A; pad0/pad4: one `LDS.128` | pad0/pad4 ✓ (80 LDS, 16 × `.128`). **pad1: 92 LDS, not 128** | **Miss, and the best lesson** (below) |
| P1b | pad4 < pad1 ≲ pad0 in time; wavefronts 88 / 132 / 144 | Time 114 < 119 < 138 µs ✓. Wavefronts 108 / 116 / 164 | Ranking ✓, numbers ✗ → corrected model matches exactly |
| P1c | FMA-pipe rank pad4 > reg64 > reg32 > tiled > naive | 48.1 > 45.1 > 28.3 > 14.8 > 13.5 | ✓ |
| P2 | FFMA = M·N·K for all; reg64 LDS = 2,097,152; conflicts pad0 ~983k, pad4 ~66k, pad1 ≈ 0 | FFMA **134,217,728 for all 8 kernels**. LDS static×trips = measured **exactly** for all 8. pad0 **983,040** and pad4 **65,536**, both exact. pad1 65,536 (not 0) | ✓ except pad1 conflicts, explained below |
| P3 | 64–96 regs; spills below ~48; occupancy > no-spill point buys < 10%; wave quantization step | 72 regs (3 blocks/SM) ✓. Spills at 32 and 40 ✓. 75% → 100% occupancy: ≤ 2% ✓. **No wave-quantization step** ✗ | Mostly ✓; quantization miss explained below |
| P4 | DRAM not the limit; ~15% of peak; clock below boost | DRAM ≈ 5.2 MB / 119 µs ≈ 44 GB/s ≈ 14% ✓. SM clock 0.93–1.14 GHz vs 1.59 boost ✓. DRAM read 2.2 MB vs 1.3 MB predicted ✗ (unexplained) | Mostly ✓ |
| P5 | naive long_scoreboard; tiled barrier + MIO; reg64 less short-scoreboard; M=32 latency-bound | tiled **mio_throttle 21.7** → reg32 10.5 → reg64 4.75 ✓. naive is **lg_throttle 17.7** (global-load queue full), not waiting on data ✗. Barrier never in the top 6 ✗. M=32: SOL 8.7% (reg64), 8.9% (cuBLAS), long_scoreboard top ✓ | Partial |
| P6 | cuBLAS `*sgemm*`, ≥128 regs, ≤25% occupancy, wins on ILP | `volta_sgemm_128x64_nn`, 122 regs, 50% theoretical / 32.8% achieved, **FMA pipe 80%** vs my best 48% | Direction ✓, numbers ✗ |

### Finding 1: the compiler was smarter than my model (P1a)
With PAD = 1 the A-tile row is 65 floats = 260 B, so only every 4th row starts 16-byte aligned. I predicted four scalar loads per row. ptxas instead covered each row with the widest aligned loads it could, **including reading the padding word**:

```
kk=0 (offset 0):     LDS.U.128 [R28.X16]            ; aligned
kk=1 (offset 260):   LDS.U.128 [R28.X16+0x100]      ; starts 4 B early: pad word + 3 floats
                     LDS.U     [R28.X16+0x110]      ; the 4th float
kk=2 (offset 520):   LDS.U.64  [+0x208] , LDS.U.64 [+0x210]
kk=3 (offset 780):   LDS.U     [+0x30c] , LDS.U.128 [+0x310]
```
That is 1 + 2 + 2 + 2 = **7 instructions per 4 rows instead of 16**, so 28 A-loads + 64 W-loads = **92** per k-tile, the exact count ncu measured (1,507,328 = 256 blocks × 8 warps × 8 k-tiles × 92). The prediction from the source was wrong; the SASS was right, and the counter agreed with it to the instruction.

### Finding 2: the bank-conflict model, corrected, predicts the counter exactly
Two things my P1b model got wrong, both found by the counters:
1. **pad1 stores are 2-way conflicted, not conflict-free.** A warp stores two rows r and r+1 (threads 0–15 and 16–31). Bank = (c + r) mod 32 for the first half and (c + r + 1) mod 32 for the second, which overlap in 15 banks. 4 stores × 1 extra wavefront × 256 × 8 × 8 = **65,536**, exactly what ncu counted.
2. **Each `LDS.128` costs 2 wavefronts here, `.64` and scalar cost 1.** (Hypothesis for why: 128-bit shared loads are processed per half-warp.)

Corrected model, wavefronts per warp per k-tile = LDS + extra for `.128` + STS + store conflicts:

| | LDS | + `.128` extra | + STS | + conflicts | = model | ncu |
|---|---|---|---|---|---|---|
| reg64 (pad1) | 92 | 12 | 8 | 4 | **116** | **116** |
| reg64_pad4 | 80 | 16 | 8 | 4 | **108** | **108** |
| reg64_pad0 | 80 | 16 | 8 | 60 | **164** | **164** |
| tiled | 40 | 8 | 2 | 0 | **50** | **50** |

And time follows wavefronts: 108 → 114 µs, 116 → 119 µs, 164 → 138 µs. Shared-memory traffic is the limiter for my kernels (LSU 67–87% busy, top stall `mio_throttle`).

### Finding 3: static STS ≠ dynamic STS
Static SASS has 14 `STS` in the reg64 loop; only 8 per thread execute (ncu: 131,072). The tile-load loops `for (i = tid; i < BM*BK; i += NT)` were compiled into a fast path (4 + 4 back-to-back `STS`) and a general path that never runs here. **Static counts need control flow; a counter tells you which path actually ran.**

### Finding 4: occupancy didn't matter; wave quantization didn't either
- `__launch_bounds__(256, 3)` got 64 registers → 4 blocks/SM (100%), and achieved occupancy rose 65% → 80%. Time: **119 → 118 µs**.
- `-maxrregcount` sweep at M=8192: 48–64 regs (100%) = 475–483 µs; 80 regs (75%) = 482–487 µs. Flat.
- I predicted 2.13 → 1.6 waves would show a step. It didn't, because blocks aren't independent: when the **LSU pipe is the bottleneck, 4 resident blocks share the same pipe that 3 did**. Total work per SM is the same, so time is the same. Wave quantization matters when blocks are latency-bound, not throughput-bound.
- **Spills** are what hurt: at 32 regs, 40–48 B of local memory and 19–25 `STL`/`LDL` in the SASS → **+19%** (reg64, 574 vs 482 µs) and **+23%** (pad4) at M=8192. At 40 regs, 8–16 B spill → +1–2%.

### Finding 5: what cuBLAS does differently
Fewer warps, more work each. ncu launch stats: 128 threads per block, 128 blocks (a 128×64 tile each), **0.8 waves**, 122 regs, 50% theoretical / 33% achieved occupancy. That is **64 outputs per thread vs my 16**, so far more FMAs per shared-memory load: **FMA pipe 80%**, LSU only 43%. It doesn't even fill one wave and still wins. *Next optimization for my kernel: an 8×8 per-thread tile with vectorized W loads.* (I didn't read cuBLAS's own SASS in this pass.)

### Finding 6: M=32 is a different problem
At one sentence, reg64 is the **slowest** (17 µs vs naive 8.5, cuBLAS 9.6): 8 blocks on 40 SMs, SM throughput 8.7%, top stall long_scoreboard. Same for cuBLAS (8.9%). This confirms the original project's conclusion: batch or remove launch overhead (CUDA graphs); no register or bank tweak fixes an empty GPU.

### Measurement problem I found in my own harness
In the register sweep, **identical SASS** (r80, r96, r128 all build 80 regs and the same 666-instruction loop) gave 165, 160 and 227 µs for reg64 at M=2048, with CV 0.0% inside a run. pad4 in the same processes was stable at ~120 µs, and M=8192 (longer runs) was stable for both. The pattern: reg64 is the **first** kernel timed after the ~1 s CPU reference. Hypothesis: the GPU clock was still ramping (10 warm-up launches is only ~1 ms). So the M=2048 reg64 sweep points are **not used**; conclusions come from M=8192 and the in-process MINB variants. Fix in the harness: `--warm-ms 300`, a timed warm-up. Rerun only that stage with `STAGES=3`.

### Not explained yet
- `-maxrregcount=32` **with** `__launch_bounds__(256)` gave 64 registers: neither 32 (flag obeyed) nor 72 (flag ignored). Now captured in `resource_usage_lb_r32.txt` on the next run.
- DRAM read 2.2 MB vs the 1.3 MB of A + W.
- Small fractional bank conflicts on reg32 and the MINB variants (not in my model).
