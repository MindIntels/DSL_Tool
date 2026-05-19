"""
Normalization Kernels in Triton: RMSNorm, LayerNorm, Gemma-style
"""
import torch
import triton
import triton.language as tl
import time


@triton.jit
def rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_x, N,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)

    # RMS
    x_sq = x * x
    mean_sq = tl.sum(x_sq, axis=0) / N
    rms_inv = 1.0 / tl.sqrt(mean_sq + eps)

    y = x * rms_inv * w
    tl.store(Y_ptr + row * stride_x + cols, y, mask=mask)


@triton.jit
def layernorm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    stride_x, N,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Mean and variance
    mean = tl.sum(x, axis=0) / N
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    y = x_centered * rstd * w + b
    tl.store(Y_ptr + row * stride_x + cols, y, mask=mask)


@triton.jit
def gemma_rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_x, N,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Gemma-style: y = (1 + w) * x / rms(x)"""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    x_sq = x * x
    mean_sq = tl.sum(x_sq, axis=0) / N
    rms_inv = 1.0 / tl.sqrt(mean_sq + eps)

    y = (1.0 + w) * x * rms_inv
    tl.store(Y_ptr + row * stride_x + cols, y, mask=mask)


@triton.jit
def fused_residual_rmsnorm_kernel(
    X_ptr, Res_ptr, W_ptr, Xout_ptr, Y_ptr,
    stride_x, N,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused residual add + RMSNorm"""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(Res_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)

    # Add residual
    x = x + r
    tl.store(Xout_ptr + row * stride_x + cols, x, mask=mask)

    # RMSNorm
    x_sq = x * x
    mean_sq = tl.sum(x_sq, axis=0) / N
    rms_inv = 1.0 / tl.sqrt(mean_sq + eps)
    y = x * rms_inv * w
    tl.store(Y_ptr + row * stride_x + cols, y, mask=mask)


# ---- Python Wrappers ----
def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    batch = x.shape[0]
    N = x.shape[-1]
    y = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(N)
    rmsnorm_kernel[(batch,)](x, weight, y, x.stride(0), N, eps=eps, BLOCK_N=BLOCK_N)
    return y


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    batch = x.shape[0]
    N = x.shape[-1]
    y = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(N)
    layernorm_kernel[(batch,)](x, weight, bias, y, x.stride(0), N, eps=eps, BLOCK_N=BLOCK_N)
    return y


def gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    batch = x.shape[0]
    N = x.shape[-1]
    y = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(N)
    gemma_rmsnorm_kernel[(batch,)](x, weight, y, x.stride(0), N, eps=eps, BLOCK_N=BLOCK_N)
    return y


# ---- Benchmark & Test ----
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda"
    batch, hidden = 32, 4096

    print("=" * 60)
    print("Triton Normalization Kernels")
    print("=" * 60)

    x = torch.randn(batch, hidden, device=device, dtype=torch.float32)
    w = torch.ones(hidden, device=device, dtype=torch.float32)
    b = torch.zeros(hidden, device=device, dtype=torch.float32)

    # RMSNorm
    y = rmsnorm(x, w)
    rms = (x ** 2).mean(dim=-1, keepdim=True).sqrt()
    y_ref = x / (rms + 1e-6) * w
    err = (y - y_ref).abs().max().item()
    print(f"\n[RMSNorm] batch={batch}, hidden={hidden}")
    print(f"  Max error: {err:.2e} -> {'PASS' if err < 1e-4 else 'FAIL'}")

    # LayerNorm
    y_ln = layernorm(x, w, b)
    y_ref_ln = torch.nn.functional.layer_norm(x, [hidden], w, b)
    err_ln = (y_ln - y_ref_ln).abs().max().item()
    print(f"\n[LayerNorm] batch={batch}, hidden={hidden}")
    print(f"  Max error: {err_ln:.2e} -> {'PASS' if err_ln < 1e-4 else 'FAIL'}")

    # Gemma RMSNorm
    w_gemma = torch.zeros(hidden, device=device, dtype=torch.float32)
    y_gemma = gemma_rmsnorm(x, w_gemma)
    y_ref_gemma = x / (rms + 1e-6)  # (1+0) * x / rms
    err_gemma = (y_gemma - y_ref_gemma).abs().max().item()
    print(f"\n[Gemma RMSNorm] batch={batch}, hidden={hidden}")
    print(f"  Max error: {err_gemma:.2e} -> {'PASS' if err_gemma < 1e-4 else 'FAIL'}")

    # Benchmark
    torch.cuda.synchronize()
    nIter = 200
    t0 = time.time()
    for _ in range(nIter):
        rmsnorm(x, w)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / nIter * 1000
    gbps = 2 * batch * hidden * 4 / (ms * 1e-3) / 1e9
    print(f"\n[RMSNorm Perf] {ms:.4f} ms | {gbps:.1f} GB/s")

    t0 = time.time()
    for _ in range(nIter):
        layernorm(x, w, b)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / nIter * 1000
    print(f"[LayerNorm Perf] {ms:.4f} ms | {2*batch*hidden*4/(ms*1e-3)/1e9:.1f} GB/s")

    print("\n" + "=" * 60)
