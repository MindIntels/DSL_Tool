"""
BF16 GEMM using Triton
C = A @ B where A: [M,K] bf16, B: [K,N] bf16, C: [M,N] fp32
"""
import torch
import triton
import triton.language as tl
import time


@triton.jit
def bf16_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to first block of A and B
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # Load with boundary check
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Store
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def bf16_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.dtype == torch.bfloat16 and B.dtype == torch.bfloat16
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    bf16_gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C


@triton.jit
def fp8_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    scale_a, scale_b,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Apply scaling
    acc = acc * scale_a * scale_b

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def fp8_gemm(A: torch.Tensor, B: torch.Tensor, scale_a: float, scale_b: float) -> torch.Tensor:
    """FP8 GEMM with per-tensor scaling."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    fp8_gemm_kernel[grid](
        A, B, C, scale_a, scale_b,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C


@triton.jit
def grouped_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K, num_groups,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Batched/Grouped GEMM - each group is a separate matmul."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    group_id = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Offset by group
    group_offset_a = group_id * M * K
    group_offset_b = group_id * K * N
    group_offset_c = group_id * M * N

    a_ptrs = A_ptr + group_offset_a + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + group_offset_b + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + group_offset_c + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def grouped_gemm(A: torch.Tensor, B: torch.Tensor, num_groups: int) -> torch.Tensor:
    """A: [num_groups, M, K], B: [num_groups, K, N] -> C: [num_groups, M, N]"""
    _, M, K = A.shape
    _, K2, N = B.shape
    assert K == K2
    C = torch.empty((num_groups, M, N), device=A.device, dtype=A.dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), num_groups)

    grouped_gemm_kernel[grid](
        A, B, C,
        M, N, K, num_groups,
        A.stride(1), A.stride(2),
        B.stride(1), B.stride(2),
        C.stride(1), C.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C


# ---- Benchmark & Test ----
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda"

    print("=" * 60)
    print("Triton GEMM Kernels Benchmark")
    print("=" * 60)

    # BF16 GEMM
    M, N, K = 4096, 4096, 4096
    A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    B = torch.randn(K, N, device=device, dtype=torch.bfloat16)

    C = bf16_gemm(A, B)
    C_ref = (A.float() @ B.float())
    err = (C - C_ref).abs().max().item()
    print(f"\n[BF16 GEMM] M={M}, N={N}, K={K}")
    print(f"  Max error: {err:.2e} -> {'PASS' if err < 1.0 else 'FAIL'}")

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        bf16_gemm(A, B)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / 20 * 1000
    tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
    print(f"  Avg time: {ms:.3f} ms | {tflops:.2f} TFLOPS")

    # FP8 (using bf16 as proxy since FP8 needs H100)
    print(f"\n[FP8 GEMM] M={M}, N={N}, K={K} (bf16 proxy)")
    A_fp8 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    B_fp8 = torch.randn(K, N, device=device, dtype=torch.bfloat16)
    C_fp8 = fp8_gemm(A_fp8, B_fp8, 1.0, 1.0)
    C_ref = (A_fp8.float() @ B_fp8.float())
    err = (C_fp8 - C_ref).abs().max().item()
    print(f"  Max error: {err:.2e} -> {'PASS' if err < 1.0 else 'FAIL'}")

    # Grouped GEMM
    num_groups, M_g, N_g, K_g = 8, 512, 512, 256
    A_g = torch.randn(num_groups, M_g, K_g, device=device, dtype=torch.bfloat16)
    B_g = torch.randn(num_groups, K_g, N_g, device=device, dtype=torch.bfloat16)
    C_g = grouped_gemm(A_g, B_g, num_groups)
    C_ref = torch.bmm(A_g.float(), B_g.float()).bfloat16()
    err = (C_g.bfloat16() - C_ref).abs().max().item()
    print(f"\n[Grouped GEMM] groups={num_groups}, M={M_g}, N={N_g}, K={K_g}")
    print(f"  Max error: {err:.2e} -> {'PASS' if err < 1.0 else 'FAIL'}")

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        grouped_gemm(A_g, B_g, num_groups)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / 20 * 1000
    tflops = 2 * num_groups * M_g * N_g * K_g / (ms * 1e-3) / 1e12
    print(f"  Avg time: {ms:.3f} ms | {tflops:.2f} TFLOPS")

    print("\n" + "=" * 60)
    print("All GEMM tests complete.")
