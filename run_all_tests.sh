#!/bin/bash
# run_all_tests.sh - One-click test runner for all Transformer kernels
# Usage: bash run_all_tests.sh [cuda|triton|tilelang|all]

set -e
cd "$(dirname "$0")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       Transformer Kernels - Complete Test Suite              ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

MODE=${1:-all}

# ============ CUDA Tests ============
run_cuda_tests() {
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  CUDA Kernel Tests${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Check for nvcc
    if ! command -v nvcc &> /dev/null; then
        echo -e "${RED}nvcc not found. Skipping CUDA tests.${NC}"
        return
    fi

    # Detect GPU arch
    GPU_ARCH="sm_80"
    if nvidia-smi &> /dev/null; then
        COMPUTE=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
        if [ ! -z "$COMPUTE" ]; then
            GPU_ARCH="sm_${COMPUTE}"
        fi
    fi
    echo -e "  GPU Architecture: ${GREEN}${GPU_ARCH}${NC}"
    echo ""

    mkdir -p build/gemm build/normalization build/activations build/rope build/qwen3next

    CUDA_FILES=(
        "cuda/gemm/bf16_gemm.cu"
        "cuda/gemm/grouped_gemm.cu"
        "cuda/normalization/rmsnorm.cu"
        "cuda/normalization/layernorm.cu"
        "cuda/normalization/gemma_norm.cu"
        "cuda/activations/fused_gated_activations.cu"
        "cuda/rope/rope.cu"
        "cuda/qwen3next/gated_softmax_attention.cu"
        "cuda/qwen3next/zero_centered_rmsnorm.cu"
        "cuda/qwen3next/gated_delta_rule.cu"
    )

    PASS=0
    FAIL=0

    for src in "${CUDA_FILES[@]}"; do
        bin="build/${src%.cu}"
        bin="${bin#cuda/}"
        mkdir -p "$(dirname "build/$bin")"
        echo -ne "  Compiling ${src}... "
        if nvcc -O3 -std=c++17 -arch=${GPU_ARCH} --use_fast_math -o "build/${bin}" "$src" 2>/dev/null; then
            echo -e "${GREEN}OK${NC}"
            echo -ne "  Running... "
            if timeout 30 "./build/${bin}" 2>/dev/null | tail -3; then
                PASS=$((PASS + 1))
            else
                echo -e "${RED}RUNTIME ERROR${NC}"
                FAIL=$((FAIL + 1))
            fi
        else
            echo -e "${RED}COMPILE ERROR${NC}"
            FAIL=$((FAIL + 1))
        fi
        echo ""
    done

    echo -e "  CUDA Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}"
}

# ============ Triton Tests ============
run_triton_tests() {
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  Triton Kernel Tests${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    if ! python3 -c "import triton" 2>/dev/null; then
        echo -e "${RED}Triton not installed. Install with: pip install triton${NC}"
        return
    fi

    TRITON_FILES=(
        "triton/gemm/gemm_kernels.py"
        "triton/normalization/norm_kernels.py"
        "triton/activations/activation_kernels.py"
        "triton/rope/rope_kernel.py"
        "triton/qwen3next/qwen3next_kernels.py"
    )

    PASS=0
    FAIL=0

    for script in "${TRITON_FILES[@]}"; do
        echo -e "  Running ${script}..."
        if timeout 60 python3 "$script" 2>&1 | grep -E "PASS|FAIL|error|Error" | head -10; then
            PASS=$((PASS + 1))
        else
            echo -e "${RED}  ERROR${NC}"
            FAIL=$((FAIL + 1))
        fi
        echo ""
    done

    echo -e "  Triton Results: ${GREEN}${PASS} scripts${NC} executed"
}

# ============ TileLang Tests ============
run_tilelang_tests() {
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  TileLang Kernel Tests${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    TILELANG_FILES=(
        "tilelang/gemm/gemm_kernels.py"
        "tilelang/normalization/norm_kernels.py"
        "tilelang/activations/activation_kernels.py"
        "tilelang/rope/rope_kernel.py"
        "tilelang/qwen3next/qwen3next_kernels.py"
    )

    PASS=0
    FAIL=0

    for script in "${TILELANG_FILES[@]}"; do
        echo -e "  Running ${script}..."
        if timeout 60 python3 "$script" 2>&1 | grep -E "PASS|FAIL|error|Error" | head -10; then
            PASS=$((PASS + 1))
        else
            echo -e "${RED}  ERROR${NC}"
            FAIL=$((FAIL + 1))
        fi
        echo ""
    done

    echo -e "  TileLang Results: ${GREEN}${PASS} scripts${NC} executed"
}

# ============ Main ============
case "$MODE" in
    cuda)
        run_cuda_tests
        ;;
    triton)
        run_triton_tests
        ;;
    tilelang)
        run_tilelang_tests
        ;;
    all)
        run_cuda_tests
        run_triton_tests
        run_tilelang_tests
        ;;
    *)
        echo "Usage: $0 [cuda|triton|tilelang|all]"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    TEST SUITE COMPLETE                       ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
