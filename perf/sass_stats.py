#!/usr/bin/env python3
"""Count SASS instructions per GEMM kernel, for the whole kernel and its main loop.

    cuobjdump -sass bin/perf_gemm > sass.txt
    python3 perf/sass_stats.py sass.txt --tag default --out results/perf

The main loop is the backward branch (BRA to a lower address) whose body has the
most FFMAs: for the tiled/regblock kernels that is one k-tile (global loads,
STS into shared memory, barrier, the unrolled kk loop, barrier).

Writes <out>/sass_summary.csv (one row per kernel and tag) and
<out>/sass_loops/<tag>_<kernel>.txt (the main loop listing, for reading).
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
from collections import Counter

FUNC_RE = re.compile(r"^\s*Function\s*:\s*(\S+)")
INSTR_RE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")
BRA_RE = re.compile(r"\bBRA\b[^;]*?(0x[0-9a-f]+)")

# Instruction classes we care about. Key = column name, value = test on the opcode.
CLASSES = {
    "ffma": lambda op: op.split(".")[0] == "FFMA",
    "lds": lambda op: op.split(".")[0] == "LDS",
    "lds128": lambda op: op.split(".")[0] == "LDS" and ".128" in op,
    "lds64": lambda op: op.split(".")[0] == "LDS" and ".64" in op,
    "sts": lambda op: op.split(".")[0] == "STS",
    "ldg": lambda op: op.split(".")[0] == "LDG",
    "stg": lambda op: op.split(".")[0] == "STG",
    "bar": lambda op: op.split(".")[0] == "BAR",
    "ldl": lambda op: op.split(".")[0] == "LDL",  # local-memory load = register spill reload
    "stl": lambda op: op.split(".")[0] == "STL",  # local-memory store = register spill
}


def demangle(names):
    exe = shutil.which("c++filt") or shutil.which("llvm-cxxfilt")
    if not exe:
        return {n: n for n in names}
    out = subprocess.run([exe], input="\n".join(names), capture_output=True, text=True).stdout.splitlines()
    return dict(zip(names, out)) if len(out) == len(names) else {n: n for n in names}


def short_name(demangled):
    """Map a demangled kernel to the perf_gemm kernel name."""
    if "gemm_naive_kernel" in demangled:
        return "naive"
    if "gemm_tiled_kernel" in demangled:
        return "tiled"
    m = re.search(r"gemm_regblock_kernel<([^>]*)>", demangled)
    args = [int(x) for x in re.findall(r"-?\d+", m.group(1))] if m else None
    if args is None:  # c++filt missing or failed: read the Itanium mangling, ...kernelILi64ELi64E...
        m = re.search(r"gemm_regblock_kernelI((?:Li\d+E)+)E", demangled)
        args = [int(x) for x in re.findall(r"Li(\d+)E", m.group(1))] if m else None
    if args:
        bm, bn, bk, tm, tn = args[:5]
        pad = args[5] if len(args) > 5 else 1
        minb = args[6] if len(args) > 6 else 1
        if (bm, tm) == (32, 2):
            return "reg32" if pad == 1 and minb == 1 else f"reg32_p{pad}_m{minb}"
        if minb != 1:
            return f"reg64_minb{minb}"
        return "reg64" if pad == 1 else f"reg64_pad{pad}"
    return None


def opcode(text):
    toks = text.split()
    if toks and toks[0].startswith("@"):  # predicate guard, e.g. @P0 or @!PT
        toks = toks[1:]
    return toks[0] if toks else ""


def parse(path):
    funcs, cur = {}, None
    with open(path) as f:
        for line in f:
            m = FUNC_RE.match(line)
            if m:
                cur = m.group(1)
                funcs[cur] = []
                continue
            if cur is None:
                continue
            m = INSTR_RE.search(line)
            if m:
                funcs[cur].append((int(m.group(1), 16), m.group(2)))
    return funcs


def count(instrs):
    c = Counter()
    for _, text in instrs:
        op = opcode(text)
        for k, test in CLASSES.items():
            if test(op):
                c[k] += 1
    c["total"] = len(instrs)
    return c


def main_loop(instrs):
    """Return the backward-branch loop body with the most FFMAs (ties: shortest)."""
    best = None
    for addr, text in instrs:
        if opcode(text).split(".")[0] != "BRA":
            continue
        m = BRA_RE.search(text)
        if not m:
            continue
        target = int(m.group(1), 16)
        if target >= addr:
            continue
        body = [(a, t) for a, t in instrs if target <= a <= addr]
        key = (count(body)["ffma"], -len(body))
        if best is None or key > best[0]:
            best = (key, body)
    return best[1] if best else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sass")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--out", default="results/perf")
    a = ap.parse_args()

    funcs = parse(a.sass)
    names = demangle(list(funcs))
    os.makedirs(os.path.join(a.out, "sass_loops"), exist_ok=True)
    path = os.path.join(a.out, "sass_summary.csv")
    fresh = not os.path.exists(path)
    cols = ["total"] + list(CLASSES)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(["tag", "kernel"] + cols + ["loop_" + c for c in cols] + ["loop_ffma_per_lds"])
        for mangled, instrs in funcs.items():
            k = short_name(names[mangled])
            if k is None:
                continue
            whole, body = count(instrs), main_loop(instrs)
            lc = count(body)
            ratio = round(lc["ffma"] / lc["lds"], 2) if lc["lds"] else ""
            w.writerow([a.tag, k] + [whole[c] for c in cols] + [lc[c] for c in cols] + [ratio])
            with open(os.path.join(a.out, "sass_loops", f"{a.tag}_{k}.txt"), "w") as g:
                g.write(f"// {names[mangled]}\n// main loop: {len(body)} instructions, "
                        f"{lc['ffma']} FFMA, {lc['lds']} LDS ({lc['lds128']} x .128), {lc['sts']} STS, "
                        f"{lc['ldg']} LDG, {lc['bar']} BAR, {lc['ldl']} LDL / {lc['stl']} STL\n")
                for addr, text in body:
                    g.write(f"/*{addr:04x}*/  {text}\n")
            print(f"{a.tag:>8} {k:<12} total {whole['total']:4}  loop {len(body):4} instr: "
                  f"FFMA {lc['ffma']:3}  LDS {lc['lds']:3} (.128 {lc['lds128']:2})  STS {lc['sts']:2}  "
                  f"LDG {lc['ldg']:2}  BAR {lc['bar']}  spills LDL {whole['ldl']} STL {whole['stl']}  "
                  f"FFMA/LDS {ratio}")


if __name__ == "__main__":
    main()
