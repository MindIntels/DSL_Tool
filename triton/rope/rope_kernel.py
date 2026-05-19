"""
RoPE (Rotary Position Embedding) in Triton - LLaMA style including LLaMA 3.1
"""
import torch
import triton
import triton.language as tl
import time
import math


@triton.jit
def rope_kernel(
    X_ptr, Out_ptr,
    seq_len, num_heads, head_dim,
    stride_b, stride_s, stride_h, stride_d,
    base: tl.constexpr,
    HALF_DIM: tl.constexpr,
):
    """Standard RoPE: apply rotation to pairs [x_i, x_{i+half}]"""
    batch_idx = tl.program_id(2)
    seq_idx = tl.program_id(1)
    head_idx = tl.program_id(0)

    pos = seq_idx  # position = sequence index
    offs_d = tl.arange(0, HALF_DIM)
    mask = offs_d < head_dim // 2

    # Compute frequencies: theta_i = pos / base^(2i/d)
    # tl.math.pow not available in Triton 3.x; use exp(log) instead
    log_base = tl.log(tl.full([1], base, dtype=tl.float32))
    exponent = (2.0 * offs_d.to(tl.float32)) / head_dim
    freq = 1.0 / tl.exp(exponent * log_base)
    theta = pos * freq
    cos_theta = tl.cos(theta)
    sin_theta = tl.sin(theta)

    # Load x0 and x1
    offset_base = batch_idx * stride_b + seq_idx * stride_s + head_idx * stride_h
    x0 = tl.load(X_ptr + offset_base + offs_d * stride_d, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(X_ptr + offset_base + (offs_d + head_dim // 2) * stride_d, mask=mask, other=0.0).to(tl.float32)

    # Apply rotation
    y0 = x0 * cos_theta - x1 * sin_theta
    y1 = x0 * sin_theta + x1 * cos_theta

    tl.store(Out_ptr + offset_base + offs_d * stride_d, y0, mask=mask)
    tl.store(Out_ptr + offset_base + (offs_d + head_dim // 2) * stride_d, y1, mask=mask)


@triton.jit
def rope_llama31_kernel(
    X_ptr, Out_ptr,
    seq_len, num_heads, head_dim,
    stride_b, stride_s, stride_h, stride_d,
    base: tl.constexpr,
    scaling_factor: tl.constexpr,
    low_freq_factor: tl.constexpr,
    high_freq_factor: tl.constexpr,
    original_max_pos: tl.constexpr,
    HALF_DIM: tl.constexpr,
):
    """LLaMA 3.1 RoPE with frequency-based scaling."""
    batch_idx = tl.program_id(2)
    seq_idx = tl.program_id(1)
    head_idx = tl.program_id(0)

    pos = seq_idx
    offs_d = tl.arange(0, HALF_DIM)
    mask = offs_d < head_dim // 2

    # Base frequencies (use exp(log) since tl.math.pow not in Triton 3.x)
    log_base = tl.log(tl.full([1], base, dtype=tl.float32))
    exponent = (2.0 * offs_d.to(tl.float32)) / head_dim
    base_freq = 1.0 / tl.exp(exponent * log_base)
    wavelength = 2.0 * 3.14159265358979 / base_freq

    old_ctx = original_max_pos * 1.0
    low_bound = low_freq_factor * old_ctx
    high_bound = high_freq_factor * old_ctx

    # Three regions: high freq (scaled), transition, low freq (unscaled)
    # High freq region: wavelength < low_bound
    is_high_freq = wavelength < low_bound
    # Low freq region: wavelength > high_bound
    is_low_freq = wavelength > high_bound

    # Smooth interpolation for transition band
    smooth = (old_ctx / wavelength - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smooth = tl.where(smooth < 0.0, 0.0, smooth)
    smooth = tl.where(smooth > 1.0, 1.0, smooth)

    freq_scaled = base_freq / scaling_factor
    freq = tl.where(is_high_freq, freq_scaled,
                    tl.where(is_low_freq, base_freq,
                             (1.0 - smooth) * freq_scaled + smooth * base_freq))

    theta = pos * freq
    cos_theta = tl.cos(theta)
    sin_theta = tl.sin(theta)

    offset_base = batch_idx * stride_b + seq_idx * stride_s + head_idx * stride_h
    x0 = tl.load(X_ptr + offset_base + offs_d * stride_d, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(X_ptr + offset_base + (offs_d + head_dim // 2) * stride_d, mask=mask, other=0.0).to(tl.float32)

    y0 = x0 * cos_theta - x1 * sin_theta
    y1 = x0 * sin_theta + x1 * cos_theta

    tl.store(Out_ptr + offset_base + offs_d * stride_d, y0, mask=mask)
    tl.store(Out_ptr + offset_base + (offs_d + head_dim // 2) * stride_d, y1, mask=mask)


# ---- Python Wrappers ----
def apply_rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """x: [batch, seq_len, num_heads, head_dim] -> rotated x"""
    B, S, H, D = x.shape
    out = torch.empty_like(x)
    HALF_DIM = triton.next_power_of_2(D // 2)
    grid = (H, S, B)
    rope_kernel[grid](
        x, out, S, H, D,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        base=base, HALF_DIM=HALF_DIM,
    )
    return out


def apply_rope_llama31(x: torch.Tensor, base: float = 500000.0,
                        scaling_factor: float = 8.0,
                        low_freq_factor: float = 1.0,
                        high_freq_factor: float = 4.0,
                        original_max_pos: int = 8192) -> torch.Tensor:
    """LLaMA 3.1 RoPE with extended context."""
    B, S, H, D = x.shape
    out = torch.empty_like(x)
    HALF_DIM = triton.next_power_of_2(D // 2)
    grid = (H, S, B)
    rope_llama31_kernel[grid](
        x, out, S, H, D,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        base=base, scaling_factor=scaling_factor,
        low_freq_factor=low_freq_factor, high_freq_factor=high_freq_factor,
        original_max_pos=original_max_pos, HALF_DIM=HALF_DIM,
    )
    return out


# ---- Test & Benchmark ----
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda"
    batch, seq_len, num_heads, head_dim = 2, 128, 32, 128

    print("=" * 60)
    print("Triton RoPE Kernels")
    print("=" * 60)

    x = torch.randn(batch, seq_len, num_heads, head_dim, device=device, dtype=torch.float32)

    # Standard RoPE test
    out = apply_rope(x, base=10000.0)

    # CPU reference
    half = head_dim // 2
    freqs = 1.0 / (10000.0 ** (torch.arange(0, half, device=device).float() * 2 / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    theta = torch.outer(positions, freqs)  # [seq, half]
    cos_t = theta.cos()[None, :, None, :]  # [1, seq, 1, half]
    sin_t = theta.sin()[None, :, None, :]

    x0 = x[..., :half]
    x1 = x[..., half:]
    ref_0 = x0 * cos_t - x1 * sin_t
    ref_1 = x0 * sin_t + x1 * cos_t
    ref = torch.cat([ref_0, ref_1], dim=-1)

    err = (out - ref).abs().max().item()
    print(f"\n[Standard RoPE] batch={batch}, seq={seq_len}, heads={num_heads}, dim={head_dim}")
    print(f"  Max error: {err:.2e} -> {'PASS' if err < 1e-4 else 'FAIL'}")

    # LLaMA 3.1 RoPE
    out_31 = apply_rope_llama31(x)
    valid = not (torch.isnan(out_31).any() or torch.isinf(out_31).any())
    print(f"\n[LLaMA 3.1 RoPE] Output valid: {'PASS' if valid else 'FAIL'}")

    # Benchmark
    nIter = 200
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(nIter):
        apply_rope(x)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / nIter * 1000
    gbps = 2 * batch * seq_len * num_heads * head_dim * 4 / (ms * 1e-3) / 1e9
    print(f"\n[Standard RoPE Perf] {ms:.4f} ms | {gbps:.1f} GB/s")

    t0 = time.time()
    for _ in range(nIter):
        apply_rope_llama31(x)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / nIter * 1000
    print(f"[LLaMA 3.1 RoPE Perf] {ms:.4f} ms | {gbps:.1f} GB/s")

    print("\n" + "=" * 60)
