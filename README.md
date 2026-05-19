# Transformer Kernel Library

High-performance CUDA kernels for modern Transformer architectures, implemented in three frameworks:
- **CUDA C++** - Native CUDA kernels with WMMA/Tensor Core support
- **Triton** - OpenAI Triton JIT-compiled kernels  
- **TileLang** - TVM TIR-based tile programming model

## Quick Start

```bash
# Run all tests (one-click)
cd DSL_Tool
bash run_all_tests.sh

# Or selectively:
bash run_all_tests.sh cuda      # CUDA kernels only
bash run_all_tests.sh triton    # Triton kernels only
bash run_all_tests.sh tilelang  # TileLang kernels only

# Using Makefile:
make test           # All tests
make cuda           # Compile CUDA
make bench          # Benchmarks
```

## Dispatch Layer

This project now provides a **kernel dispatch management layer** so external users can call kernels through one public API using:

- **kernel name**: for example `normalization.rmsnorm`, `gemm.bf16_gemm`, `activations.swiglu`
- **implementation backend**: for example `triton` or `tilelang`

The dispatch layer resolves the requested kernel implementation and deploys execution onto the GPU platform.

### Components

- `dispatcher/registry.py`: registers available kernels by `(kernel_name, backend)`
- `dispatcher/dispatcher.py`: core runtime dispatcher and tensor serialization layer
- `dispatcher/api.py`: FastAPI service for external invocation
- `dispatcher/cli.py`: command-line interface for local testing and automation

### Programmatic API

```python
from dispatcher import KernelDispatcher

dispatcher = KernelDispatcher(device="cuda")

result = dispatcher.run_from_spec(
		kernel_name="normalization.rmsnorm",
		backend="triton",
		tensor_specs=[
				{"shape": [32, 4096], "dtype": "bfloat16", "fill": "random"},
				{"shape": [4096], "dtype": "bfloat16", "fill": "ones"},
		],
		scalar_params={"eps": 1e-6},
)

print(result.to_dict())
```

### REST API

Start service:

```bash
cd DSL_Tool
uvicorn dispatcher.api:app --host 0.0.0.0 --port 8000 --reload
```

List kernels:

```bash
curl http://127.0.0.1:8000/api/v1/kernels
```

Run a kernel:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/kernels/run \
	-H 'Content-Type: application/json' \
	-d '{
				"kernel_name": "normalization.rmsnorm",
				"backend": "triton",
				"inputs": [
					{"shape": [32, 4096], "dtype": "bfloat16", "fill": "random"},
					{"shape": [4096], "dtype": "bfloat16", "fill": "ones"}
				],
				"params": {"eps": 1e-6}
			}'
```

### CLI

```bash
cd cd DSL_Tool

# List all registered kernels
python -m dispatcher.cli list

# Inspect one kernel
python -m dispatcher.cli info --kernel normalization.rmsnorm --backend triton

# Run one kernel
python -m dispatcher.cli run \
	--kernel normalization.rmsnorm \
	--backend triton \
	--input '{"shape":[32,4096],"dtype":"bfloat16","fill":"random"}' \
	--input '{"shape":[4096],"dtype":"bfloat16","fill":"ones"}' \
	--param eps=1e-6
```

## Prerequisites

```bash
# CUDA (required for CUDA kernels)
# nvcc with SM80+ support (Ampere or newer)

# Python packages (for Triton and TileLang)
pip install torch triton
pip install tilelang  # optional, fallback to PyTorch tiled impl
```

---

## Kernel Descriptions

### 1. GEMM & Linear Operations

| Kernel | Description | Use Case |
|--------|-------------|----------|
| **BF16 GEMM** | BF16 matrix multiply via WMMA Tensor Cores | General LLM inference/training on Ampere+ |
| **FP8 GEMM** | FP8 (E4M3) with per-tensor & groupwise scaling | High-throughput inference on Hopper (H100) |
| **FP4 GEMM** | NVFP4/MXFP4 with block-wise scaling | Ultra-low-precision on Blackwell (B200) |
| **Grouped GEMM** | Batched matmul for variable-size groups | LoRA adapters, MoE expert routing |

**Key Design Choices:**
- BF16 uses tiled WMMA with shared memory double-buffering
- FP8 supports both per-tensor (single scale) and groupwise (scale per G elements) modes
- FP4 provides software emulation with correct numerics; hardware path ready for SM100+
- Grouped GEMM uses 3D grid (M-tile × N-tile × group) for maximum parallelism

### 2. Normalization

| Kernel | Description | Formula |
|--------|-------------|---------|
| **RMSNorm** | Root Mean Square Normalization | `y = w * x / √(mean(x²) + ε)` |
| **LayerNorm** | Standard Layer Normalization | `y = w * (x - μ) / √(σ² + ε) + b` |
| **Gemma Norm** | Gemma-style fused RMSNorm | `y = (1 + w) * x / √(mean(x²) + ε)` |

**Performance Characteristics:**
- Memory-bound kernels: performance measured in GB/s
- Block-level reduction using warp shuffles (no atomics)
- Fused residual variants save one global memory pass
- Vectorized loads for higher effective bandwidth

### 3. Activations

| Kernel | Description | Formula |
|--------|-------------|---------|
| **SwiGLU** | SiLU-gated linear unit | `output = SiLU(gate) × up = (gate × σ(gate)) × up` |
| **GeGLU** | GELU-gated linear unit | `output = GELU(gate) × up` |

**Fused Design:**
- Input format: `[batch, 2×hidden]` split into gate and up projections
- Single kernel pass reads both halves and produces output
- Backward pass also fused (gradient for both gate and up in one kernel)
- GELU uses fast tanh approximation: `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`

### 4. RoPE (Rotary Position Embeddings)

| Variant | Description | Parameters |
|---------|-------------|------------|
| **Standard** | Base RoPE (LLaMA 1/2) | `base=10000` |
| **LLaMA 3.1** | Extended context with freq scaling | `base=500000, factor=8×, bands=[1,4]` |

**LLaMA 3.1 Extended RoPE:**
- Low frequencies (long wavelength > 4×ctx): keep original
- High frequencies (short wavelength < 1×ctx): divide by scaling_factor
- Transition band: smooth linear interpolation
- Supports NTK-aware scaling for 128K+ context

### 5. Qwen3NeXt Kernels

| Kernel | Description | Innovation |
|--------|-------------|------------|
| **Gated Softmax Attention** | Attention with per-head gating | `O = σ(g_h) × softmax(QK^T/√d) × V` |
| **Zero-Centered RMSNorm** | RMSNorm with zero-init weights | `y = (1+w) × x/rms(x)`, w init to 0 |
| **Gated Delta Rule** | Linear attention with gated memory | `S_t = S_{t-1} + β(v - S@k)⊗k` |
| **Gated DeltaNet** | Full block: conv + delta + gating | Combines conv1d + delta rule + output gate |

**Gated Delta Rule Details:**
```
State update:  S_t = S_{t-1} + β_t × (v_t - S_{t-1}@k_t) ⊗ k_t^T
Output:        o_t = S_t @ q_t
```
- `β_t = sigmoid(...)` controls update strength
- Linear attention complexity: O(n × d²) vs O(n² × d) for softmax attention
- Suitable for long-context generation with constant memory per token

**Gated DeltaNet Architecture:**
1. Causal conv1d (short-range patterns)
2. Q/K/V/β projection
3. Gated delta rule recurrence
4. Sigmoid output gating

---

## Performance Comparison

### GEMM (4096 × 4096 × 4096)

| Implementation | BF16 TFLOPS | Notes |
|----------------|-------------|-------|
| CUDA (WMMA) | ~50-80 | Depends on tiling strategy |
| Triton | ~70-120 | Auto-tuned, good occupancy |
| cuBLAS (ref) | ~150+ | Vendor-optimized baseline |
| TileLang/TVM | ~60-100 | Generated code quality varies |

### Normalization (batch=32, hidden=4096)

| Implementation | RMSNorm (μs) | LayerNorm (μs) | Bandwidth |
|----------------|--------------|----------------|-----------|
| CUDA | ~10-20 | ~15-25 | ~400 GB/s |
| Triton | ~8-15 | ~12-20 | ~500 GB/s |
| PyTorch (ref) | ~15-30 | ~20-40 | ~300 GB/s |

### Activations (batch=32, hidden=4096)

| Implementation | SwiGLU (μs) | GeGLU (μs) |
|----------------|-------------|-------------|
| CUDA | ~5-10 | ~5-10 |
| Triton | ~4-8 | ~4-8 |
| PyTorch (ref) | ~10-20 | ~10-20 |

### RoPE (batch=2, seq=128, heads=32, dim=128)

| Implementation | Standard (μs) | LLaMA 3.1 (μs) |
|----------------|--------------|-----------------|
| CUDA | ~30-50 | ~35-60 |
| Triton | ~25-45 | ~30-50 |

*Note: Actual performance depends on GPU model, driver version, and memory configuration.*

---

## Directory Structure

```
Transformer/
├── cuda/                          # Native CUDA C++ kernels
│   ├── gemm/
│   │   ├── bf16_gemm.cu          # BF16 WMMA GEMM
│   │   ├── fp8_gemm.cu           # FP8 with scaling
│   │   ├── fp4_gemm.cu           # FP4 NVFP4/MXFP4
│   │   └── grouped_gemm.cu       # Batched/LoRA GEMM
│   ├── normalization/
│   │   ├── rmsnorm.cu
│   │   ├── layernorm.cu
│   │   └── gemma_norm.cu
│   ├── activations/
│   │   └── fused_gated_activations.cu   # SwiGLU + GeGLU
│   ├── rope/
│   │   └── rope.cu               # Standard + LLaMA 3.1
│   └── qwen3next/
│       ├── gated_softmax_attention.cu
│       ├── zero_centered_rmsnorm.cu
│       └── gated_delta_rule.cu    # Delta Rule + DeltaNet
│
├── triton/                        # OpenAI Triton kernels
│   ├── gemm/gemm_kernels.py
│   ├── normalization/norm_kernels.py
│   ├── activations/activation_kernels.py
│   ├── rope/rope_kernel.py
│   └── qwen3next/qwen3next_kernels.py
│
├── tilelang/                      # TileLang/TVM kernels
│   ├── gemm/gemm_kernels.py
│   ├── normalization/norm_kernels.py
│   ├── activations/activation_kernels.py
│   ├── rope/rope_kernel.py
│   └── qwen3next/qwen3next_kernels.py
│
├── Makefile                       # Build system
├── run_all_tests.sh              # One-click test runner
└── README.md                      # This file
```

---

## Test Examples

### CUDA

```bash
# Compile and run individual kernel
nvcc -O3 -std=c++17 -arch=sm_80 -o test_rmsnorm cuda/normalization/rmsnorm.cu
./test_rmsnorm

# Run with custom sizes
nvcc -O3 -std=c++17 -arch=sm_80 -o test_gemm cuda/gemm/bf16_gemm.cu
./test_gemm 2048 2048 2048  # M N K
```

### Triton

```python
# Run from Python
import sys
sys.path.insert(0, 'triton/gemm')
from gemm_kernels import bf16_gemm, grouped_gemm
import torch

A = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
B = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
C = bf16_gemm(A, B)
print(f"Output shape: {C.shape}, dtype: {C.dtype}")
```

### TileLang

```python
import sys
sys.path.insert(0, 'tilelang/qwen3next')
from qwen3next_kernels import TiledGatedDeltaRule
import torch

model = TiledGatedDeltaRule(head_dim=64)
Q = torch.randn(4, 128, 64, device='cuda') * 0.1
K = torch.randn(4, 128, 64, device='cuda') * 0.1
V = torch.randn(4, 128, 64, device='cuda') * 0.1
beta = torch.randn(4, 128, device='cuda')
O = model.forward(Q, K, V, beta)
```

---

## Implementation Notes

### Numerical Precision
- BF16 GEMM accumulates in FP32 for numerical stability
- FP8/FP4 use block-wise scaling to maintain dynamic range
- All normalization kernels use FP32 for reduction/accumulation
- RoPE uses FP32 for trigonometric computations

### GPU Requirements
| Kernel | Minimum SM | Recommended |
|--------|-----------|-------------|
| BF16 GEMM (WMMA) | SM80 (A100) | SM90 (H100) |
| FP8 GEMM | SM89 (L40/4090) | SM90 (H100) |
| FP4 GEMM | SM100 (B200) | SM100 |
| All others | SM70 (V100) | SM80+ |

### Key References
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (SwiGLU/GeGLU)
- [DeltaNet: Conditional State Space Models](https://arxiv.org/abs/2310.18020)
- [LLaMA 3.1 Technical Report](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
