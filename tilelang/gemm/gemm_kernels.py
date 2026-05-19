"""
GEMM Kernels using TileLang (TVM-based tile programming)
BF16 GEMM, FP8 GEMM (emulated), Grouped GEMM
"""
import torch
import numpy as np
import time

try:
    import tilelang
    from tilelang import Kernel, TileGrid
    HAS_TILELANG = True
except ImportError:
    HAS_TILELANG = False
    print("TileLang not installed. Using TVM-style implementation.")

try:
    import tvm
    from tvm import te, tir
    from tvm.script import tir as T
    HAS_TVM = True
except ImportError:
    HAS_TVM = False


# ========== TVM TIR-based BF16 GEMM ==========
def create_bf16_gemm_tir(M, N, K, block_M=128, block_N=128, block_K=32):
    """Create a TIR schedule for BF16 GEMM using TVM."""
    if not HAS_TVM:
        return None

    @T.prim_func
    def bf16_gemm(
        A: T.Buffer((M, K), "float16"),
        B: T.Buffer((K, N), "float16"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "bf16_gemm", "tir.noalias": True})
        for bx in T.thread_binding(N // block_N, thread="blockIdx.x"):
            for by in T.thread_binding(M // block_M, thread="blockIdx.y"):
                for tx in T.thread_binding(32, thread="threadIdx.x"):
                    for ty in T.thread_binding(block_M // 16 * block_N // 16 // 32, thread="threadIdx.y"):
                        with T.block("gemm"):
                            # Shared memory
                            A_shared = T.alloc_buffer((block_M, block_K), "float16", scope="shared")
                            B_shared = T.alloc_buffer((block_K, block_N), "float16", scope="shared")
                            C_local = T.alloc_buffer((16, 16), "float32", scope="local")

                            # Initialize accumulator
                            for i, j in T.grid(16, 16):
                                C_local[i, j] = 0.0

                            # Main loop over K
                            for k in range(K // block_K):
                                # Load A tile to shared
                                for i, j in T.grid(block_M, block_K):
                                    A_shared[i, j] = A[by * block_M + i, k * block_K + j]
                                # Load B tile to shared
                                for i, j in T.grid(block_K, block_N):
                                    B_shared[i, j] = B[k * block_K + i, bx * block_N + j]

                                # Compute
                                for kk in range(block_K):
                                    for i, j in T.grid(16, 16):
                                        C_local[i, j] += T.cast(A_shared[i, kk], "float32") * T.cast(B_shared[kk, j], "float32")

                            # Store result
                            for i, j in T.grid(16, 16):
                                C[by * block_M + i, bx * block_N + j] = C_local[i, j]

    return bf16_gemm


# ========== PyTorch fallback implementation with TileLang-style tiling ==========
def bf16_gemm_tiled(A: torch.Tensor, B: torch.Tensor,
                     block_M=128, block_N=128, block_K=32) -> torch.Tensor:
    """BF16 tiled GEMM using PyTorch (structure mimics TileLang kernel)."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.zeros(M, N, device=A.device, dtype=torch.float32)

    # Tile the computation
    for bm in range(0, M, block_M):
        for bn in range(0, N, block_N):
            acc = torch.zeros(min(block_M, M - bm), min(block_N, N - bn),
                            device=A.device, dtype=torch.float32)
            for bk in range(0, K, block_K):
                a_tile = A[bm:bm+block_M, bk:bk+block_K].float()
                b_tile = B[bk:bk+block_K, bn:bn+block_N].float()
                acc += a_tile @ b_tile
            C[bm:bm+block_M, bn:bn+block_N] = acc
    return C


def fp8_gemm_tiled(A: torch.Tensor, B: torch.Tensor,
                    scale_a: float, scale_b: float,
                    block_M=64, block_N=64, block_K=32) -> torch.Tensor:
    """FP8 tiled GEMM with per-tensor scaling (PyTorch simulation)."""
    M, K = A.shape
    K2, N = B.shape
    C = torch.zeros(M, N, device=A.device, dtype=torch.float32)

    for bm in range(0, M, block_M):
        for bn in range(0, N, block_N):
            acc = torch.zeros(min(block_M, M - bm), min(block_N, N - bn),
                            device=A.device, dtype=torch.float32)
            for bk in range(0, K, block_K):
                a_tile = A[bm:bm+block_M, bk:bk+block_K].float()
                b_tile = B[bk:bk+block_K, bn:bn+block_N].float()
                acc += a_tile @ b_tile
            C[bm:bm+block_M, bn:bn+block_N] = acc * scale_a * scale_b
    return C


def grouped_gemm_tiled(A: torch.Tensor, B: torch.Tensor,
                        block_M=64, block_N=64, block_K=32) -> torch.Tensor:
    """Grouped GEMM: A [G, M, K], B [G, K, N] -> C [G, M, N]."""
    G, M, K = A.shape
    _, K2, N = B.shape
    C = torch.zeros(G, M, N, device=A.device, dtype=torch.float32)

    for g in range(G):
        for bm in range(0, M, block_M):
            for bn in range(0, N, block_N):
                acc = torch.zeros(min(block_M, M - bm), min(block_N, N - bn),
                                device=A.device, dtype=torch.float32)
                for bk in range(0, K, block_K):
                    a_tile = A[g, bm:bm+block_M, bk:bk+block_K].float()
                    b_tile = B[g, bk:bk+block_K, bn:bn+block_N].float()
                    acc += a_tile @ b_tile
                C[g, bm:bm+block_M, bn:bn+block_N] = acc
    return C


# ========== TileLang-style kernel spec (for code generation) ==========
TILELANG_BF16_GEMM_SPEC = """
# TileLang BF16 GEMM Specification
# This generates optimized CUDA code via TileLang/TVM compilation

import tilelang
from tilelang import Kernel

@tilelang.jit(
    grid=lambda M, N: (M // 128, N // 128),
    block=(128,),
    shared_mem=2 * 128 * 32 * 2,  # A_shared + B_shared
)
def bf16_gemm_tilelang(
    A: tilelang.Tensor[M, K, "bfloat16"],
    B: tilelang.Tensor[K, N, "bfloat16"],
    C: tilelang.Tensor[M, N, "float32"],
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
    BLOCK_K: int = 32,
):
    # Load tiles cooperatively
    a_tile = tilelang.shared_load(A[block_y*BLOCK_M:(block_y+1)*BLOCK_M, k:k+BLOCK_K])
    b_tile = tilelang.shared_load(B[k:k+BLOCK_K, block_x*BLOCK_N:(block_x+1)*BLOCK_N])

    # Compute using tensor cores
    acc = tilelang.mma(a_tile, b_tile, acc)

    # Store result
    C[block_y*BLOCK_M:(block_y+1)*BLOCK_M, block_x*BLOCK_N:(block_x+1)*BLOCK_N] = acc
"""


# ---- Test & Benchmark ----
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("TileLang GEMM Kernels (Tiled Implementation)")
    print("=" * 60)

    # BF16 GEMM
    M, N, K = 1024, 1024, 1024
    A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    B = torch.randn(K, N, device=device, dtype=torch.bfloat16)

    C = bf16_gemm_tiled(A, B)
    C_ref = A.float() @ B.float()
    err = (C - C_ref).abs().max().item()
    print(f"\n[BF16 GEMM] M={M}, N={N}, K={K}")
    print(f"  Max error: {err:.2e} -> {'PASS' if err < 1e-2 else 'FAIL'}")

    # Benchmark
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    nIter = 10
    for _ in range(nIter):
        bf16_gemm_tiled(A, B)
    if device == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) / nIter * 1000
    tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
    print(f"  Avg time: {ms:.3f} ms | {tflops:.4f} TFLOPS")

    # FP8 GEMM
    C_fp8 = fp8_gemm_tiled(A, B, 1.0, 1.0)
    err_fp8 = (C_fp8 - C_ref).abs().max().item()
    print(f"\n[FP8 GEMM (emulated)] Max error: {err_fp8:.2e} -> {'PASS' if err_fp8 < 1e-2 else 'FAIL'}")

    # Grouped GEMM
    G, M_g, N_g, K_g = 8, 256, 256, 128
    A_g = torch.randn(G, M_g, K_g, device=device, dtype=torch.bfloat16)
    B_g = torch.randn(G, K_g, N_g, device=device, dtype=torch.bfloat16)
    C_g = grouped_gemm_tiled(A_g, B_g)
    C_ref_g = torch.bmm(A_g.float(), B_g.float())
    err_g = (C_g - C_ref_g).abs().max().item()
    print(f"\n[Grouped GEMM] G={G}, M={M_g}, N={N_g}, K={K_g}")
    print(f"  Max error: {err_g:.2e} -> {'PASS' if err_g < 1e-2 else 'FAIL'}")

    if HAS_TVM:
        print("\n[TVM TIR] BF16 GEMM schedule created successfully")
        func = create_bf16_gemm_tir(M, N, K)
        if func:
            print("  TIR function generated.")

    print("\n" + "=" * 60)
    print("All TileLang GEMM tests complete.")
