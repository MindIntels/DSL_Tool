# Makefile for CUDA Transformer Kernels
# Usage: make all | make cuda | make test_triton | make test_tilelang | make clean

NVCC := nvcc
NVCC_FLAGS := -O3 -std=c++17 -arch=sm_80 --use_fast_math
CUDA_DIR := cuda
BUILD_DIR := build

# Detect GPU architecture
GPU_ARCH ?= sm_80

# CUDA source files
CUDA_SRCS := \
	$(CUDA_DIR)/gemm/bf16_gemm.cu \
	$(CUDA_DIR)/gemm/fp8_gemm.cu \
	$(CUDA_DIR)/gemm/fp4_gemm.cu \
	$(CUDA_DIR)/gemm/grouped_gemm.cu \
	$(CUDA_DIR)/normalization/rmsnorm.cu \
	$(CUDA_DIR)/normalization/layernorm.cu \
	$(CUDA_DIR)/normalization/gemma_norm.cu \
	$(CUDA_DIR)/activations/fused_gated_activations.cu \
	$(CUDA_DIR)/rope/rope.cu \
	$(CUDA_DIR)/qwen3next/gated_softmax_attention.cu \
	$(CUDA_DIR)/qwen3next/zero_centered_rmsnorm.cu \
	$(CUDA_DIR)/qwen3next/gated_delta_rule.cu

# Build targets
CUDA_BINS := $(patsubst $(CUDA_DIR)/%.cu,$(BUILD_DIR)/%,$(CUDA_SRCS))

.PHONY: all cuda triton tilelang test clean help

help:
	@echo "============================================="
	@echo " Transformer Kernels Build System"
	@echo "============================================="
	@echo ""
	@echo "Targets:"
	@echo "  make all          - Build CUDA and run all tests"
	@echo "  make cuda         - Build all CUDA kernels"
	@echo "  make test_cuda    - Build and run CUDA tests"
	@echo "  make test_triton  - Run Triton kernel tests"
	@echo "  make test_tilelang - Run TileLang kernel tests"
	@echo "  make test         - Run all tests"
	@echo "  make bench        - Run benchmarks"
	@echo "  make clean        - Remove build artifacts"
	@echo ""
	@echo "GPU_ARCH=$(GPU_ARCH) (override with GPU_ARCH=sm_90)"

all: cuda test_triton test_tilelang

# ============ CUDA Build ============
cuda: $(CUDA_BINS)

$(BUILD_DIR)/%: $(CUDA_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -arch=$(GPU_ARCH) -o $@ $<

# ============ CUDA Tests ============
test_cuda: cuda
	@echo "============================================="
	@echo " Running CUDA Kernel Tests"
	@echo "============================================="
	@for bin in $(CUDA_BINS); do \
		echo "\n--- Running: $$bin ---"; \
		./$$bin || true; \
	done

# ============ Triton Tests ============
test_triton:
	@echo "============================================="
	@echo " Running Triton Kernel Tests"
	@echo "============================================="
	python3 triton/gemm/gemm_kernels.py
	python3 triton/normalization/norm_kernels.py
	python3 triton/activations/activation_kernels.py
	python3 triton/rope/rope_kernel.py
	python3 triton/qwen3next/qwen3next_kernels.py

# ============ TileLang Tests ============
test_tilelang:
	@echo "============================================="
	@echo " Running TileLang Kernel Tests"
	@echo "============================================="
	python3 tilelang/gemm/gemm_kernels.py
	python3 tilelang/normalization/norm_kernels.py
	python3 tilelang/activations/activation_kernels.py
	python3 tilelang/rope/rope_kernel.py
	python3 tilelang/qwen3next/qwen3next_kernels.py

# ============ All Tests ============
test: test_cuda test_triton test_tilelang
	@echo "\n============================================="
	@echo " ALL TESTS COMPLETE"
	@echo "============================================="

# ============ Benchmarks ============
bench: cuda
	@echo "============================================="
	@echo " Running Benchmarks"
	@echo "============================================="
	@echo "\n--- BF16 GEMM 4096x4096x4096 ---"
	./$(BUILD_DIR)/gemm/bf16_gemm 4096 4096 4096
	@echo "\n--- RMSNorm ---"
	./$(BUILD_DIR)/normalization/rmsnorm
	@echo "\n--- LayerNorm ---"
	./$(BUILD_DIR)/normalization/layernorm
	@echo "\n--- Activations ---"
	./$(BUILD_DIR)/activations/fused_gated_activations
	@echo "\n--- RoPE ---"
	./$(BUILD_DIR)/rope/rope
	@echo "\n--- Gated Delta Rule ---"
	./$(BUILD_DIR)/qwen3next/gated_delta_rule

# ============ Clean ============
clean:
	rm -rf $(BUILD_DIR)
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
