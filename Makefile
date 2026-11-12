# Build:   make ARCH=sm_75        (T4 = sm_75, RTX 30xx = sm_86, A100 = sm_80, L4 = sm_89)
# Test:    make test              (CUDA kernel unit tests)
# Run:     make run DATA=data     (BERT on GPU vs. CPU reference)
# Bench:   make bench
# No GPU:  make cpu-test DATA=data

NVCC     ?= nvcc
CXX      ?= c++
ARCH     ?= sm_75
DATA     ?= data
NVFLAGS  := -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Isrc -Xcompiler -Wall
CXXFLAGS := -O2 -std=c++17 -Wall -Wextra -Isrc
LIBS     := -lcublas

COMMON := src/cpu_ref.cpp src/kernels.cu
HEADERS := $(wildcard src/*.h src/*.cuh)

all: bin/bert_infer bin/test_kernels bin/bench_gemm

bin:
	mkdir -p bin results

bin/bert_infer: apps/bert_infer.cu src/bert_gpu.cu $(COMMON) $(HEADERS) | bin
	$(NVCC) $(NVFLAGS) $(filter %.cu %.cpp,$^) -o $@ $(LIBS)

bin/test_kernels: tests/test_kernels.cu $(COMMON) $(HEADERS) | bin
	$(NVCC) $(NVFLAGS) $(filter %.cu %.cpp,$^) -o $@ $(LIBS)

bin/bench_gemm: bench/bench_gemm.cu $(COMMON) $(HEADERS) | bin
	$(NVCC) $(NVFLAGS) $(filter %.cu %.cpp,$^) -o $@ $(LIBS)

bin/test_cpu: tests/test_cpu.cpp src/cpu_ref.cpp src/cpu_ref.h src/fixed_point.h | bin
	$(CXX) $(CXXFLAGS) tests/test_cpu.cpp src/cpu_ref.cpp -o $@

test: bin/test_kernels
	./bin/test_kernels

run: bin/bert_infer
	./bin/bert_infer --data $(DATA) --csv results/infer_fp32.csv
	./bin/bert_infer --data $(DATA) --fixed 18,16 --csv results/infer_fixed.csv

sweep: bin/bert_infer
	./bin/bert_infer --data $(DATA) --sweep

bench: bin/bench_gemm
	./bin/bench_gemm --csv results/bench_gemm.csv

cpu-test: bin/test_cpu
	./bin/test_cpu $(DATA)

profile: bin/bert_infer bin/bench_gemm
	./scripts/profile.sh $(DATA)

clean:
	rm -rf bin

.PHONY: all test run sweep bench cpu-test profile clean
