#!/usr/bin/env python3
"""Print the comma-separated --metrics list for ncu, keeping only names this GPU supports.

ncu stops with an error if any requested metric is unknown, and names differ
between GPU generations, so every name is checked against `ncu --query-metrics`.
"""
import subprocess
import sys

WANTED = [
    # dynamic instruction counts (compare with static SASS x trip count)
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "smsp__inst_executed.sum",
    "smsp__inst_executed_op_shared_ld.sum",
    "smsp__inst_executed_op_shared_st.sum",
    "smsp__inst_executed_op_global_ld.sum",
    "smsp__inst_executed_op_local_ld.sum",
    "smsp__inst_executed_op_local_st.sum",
    # shared memory: wavefronts and bank conflicts
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    # execution-unit utilization and occupancy
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    # clock, time, memory traffic
    "sm__cycles_elapsed.avg.per_second",
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_sector_hit_rate.pct",
]


def main():
    try:
        out = subprocess.run(["ncu", "--query-metrics"], capture_output=True, text=True, timeout=120).stdout
    except Exception as e:  # no ncu: print nothing useful, caller checks
        print(f"ncu --query-metrics failed: {e}", file=sys.stderr)
        out = ""
    known = {line.split()[0] for line in out.splitlines() if line.strip() and "__" in line.split()[0]}
    keep, drop = [], []
    for m in WANTED:
        (keep if m.split(".")[0] in known else drop).append(m)
    if drop:
        print("ncu_metrics: not available on this GPU, skipped: " + ", ".join(drop), file=sys.stderr)
    print(",".join(keep))


if __name__ == "__main__":
    main()
