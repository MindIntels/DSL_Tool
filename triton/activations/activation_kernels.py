"""
Fused Gated Activations in Triton: SwiGLU and GeGLU
"""
import torch
import triton
import triton.language as tl
import time


@triton.jit
def swiglu_kernel(
    Input_ptr, Output_ptr,
    batch_size, hidden_size,
    stride_in, stride_out,
    BLOCK: tl.constexpr,
):
    """SwiGLU: output = SiLU(gate) * up"""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    # Decode flat index to (batch, hidden)
    idx = pid * BLOCK + offs
    b = idx // hidden_size
    h = idx % hidden_size
    mask = idx < batch_size * hidden_size

    # Load gate and up from interleaved input [batch, 2*hidden]
    gate = tl.load(Input_ptr + b * stride_in + h, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(Input_ptr + b * stride_in + hidden_size + h, mask=mask, other=0.0).to(tl.float32)

    # SiLU(gate) = gate * sigmoid(gate)
    sigmoid_gate = tl.sigmoid(gate)
    silu_gate = gate * sigmoid_gate

    out = silu_gate * up
    tl.store(Output_ptr + b * stride_out + h, out, mask=mask)


@triton.jit
def geglu_kernel(
    Input_ptr, Output_ptr,
    batch_size, hidden_size,
    stride_in, stride_out,
    BLOCK: tl.constexpr,
):
    """GeGLU: output = GELU(gate) * up"""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    idx = pid * BLOCK + offs
    b = idx // hidden_size
    h = idx % hidden_size
    mask = idx < batch_size * hidden_size

    gate = tl.load(Input_ptr + b * stride_in + h, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(Input_ptr + b * stride_in + hidden_size + h, mask=mask, other=0.0).to(tl.float32)

    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # tanh(z) = 2*sigmoid(2*z) - 1  (numerically stable, no tl.math.tanh needed)
    c = 0.7978845608  # sqrt(2/pi)
    k = 0.044715
    inner = c * (gate + k * gate * gate * gate)
    tanh_val = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    gelu_gate = 0.5 * gate * (1.0 + tanh_val)

    out = gelu_gate * up
    tl.store(Output_ptr + b * stride_out + h, out, mask=mask)


@triton.jit
def swiglu_backward_kernel(
    Grad_out_ptr, Input_ptr, Grad_in_ptr,
    batch_size, hidden_size,
    stride_go, stride_in, stride_gi,
    BLOCK: tl.constexpr,
):
    """SwiGLU backward pass."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    idx = pid * BLOCK + offs
    b = idx // hidden_size
    h = idx % hidden_size
    mask = idx < batch_size * hidden_size

    go = tl.load(Grad_out_ptr + b * stride_go + h, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(Input_ptr + b * stride_in + h, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(Input_ptr + b * stride_in + hidden_size + h, mask=mask, other=0.0).to(tl.float32)

    sig = tl.sigmoid(gate)
    silu_val = gate * sig

    # d_up = SiLU(gate) * grad_output
    d_up = silu_val * go

    # d_gate = up * go * dsilu(gate)
    # dsilu = sig * (1 + gate * (1 - sig))
    dsilu = sig * (1.0 + gate * (1.0 - sig))
    d_gate = up * go * dsilu

    tl.store(Grad_in_ptr + b * stride_gi + h, d_gate, mask=mask)
    tl.store(Grad_in_ptr + b * stride_gi + hidden_size + h, d_up, mask=mask)


# ---- Python Wrappers ----
def swiglu(input: torch.Tensor) -> torch.Tensor:
    """input: [batch, 2*hidden] -> output: [batch, hidden]"""
    batch, two_hidden = input.shape
    hidden = two_hidden // 2
    output = torch.empty(batch, hidden, device=input.device, dtype=input.dtype)

    BLOCK = 1024
    n = batch * hidden
    grid = (triton.cdiv(n, BLOCK),)
    swiglu_kernel[grid](input, output, batch, hidden, input.stride(0), output.stride(0), BLOCK=BLOCK)
    return output


def geglu(input: torch.Tensor) -> torch.Tensor:
    """input: [batch, 2*hidden] -> output: [batch, hidden]"""
    batch, two_hidden = input.shape
    hidden = two_hidden // 2
    output = torch.empty(batch, hidden, device=input.device, dtype=input.dtype)

    BLOCK = 1024
    n = batch * hidden
    grid = (triton.cdiv(n, BLOCK),)
    geglu_kernel[grid](input, output, batch, hidden, input.stride(0), output.stride(0), BLOCK=BLOCK)
    return output


# ---- Test & Benchmark ----
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda"
    batch, hidden = 32, 4096

    print("=" * 60)
    print("Triton Fused Gated Activations")
    print("=" * 60)

    input_data = torch.randn(batch, 2 * hidden, device=device, dtype=torch.float32)

    # SwiGLU test
    out_swiglu = swiglu(input_data)
    gate = input_data[:, :hidden]
    up = input_data[:, hidden:]
    ref_swiglu = torch.nn.functional.silu(gate) * up
    err = (out_swiglu - ref_swiglu).abs().max().item()
    print(f"\n[SwiGLU] batch={batch}, hidden={hidden}")
    print(f"  Max error: {err:.2e} -> {'PASS' if err < 1e-5 else 'FAIL'}")

    # GeGLU test
    out_geglu = geglu(input_data)
    ref_geglu = torch.nn.functional.gelu(gate, approximate='tanh') * up
    err_geglu = (out_geglu - ref_geglu).abs().max().item()
    print(f"\n[GeGLU] batch={batch}, hidden={hidden}")
    print(f"  Max error: {err_geglu:.2e} -> {'PASS' if err_geglu < 1e-3 else 'FAIL'}")

    # Benchmark
    nIter = 200
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(nIter):
        swiglu(input_data)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / nIter * 1000
    gbps = batch * 3 * hidden * 4 / (ms * 1e-3) / 1e9
    print(f"\n[SwiGLU Perf] {ms:.4f} ms | {gbps:.1f} GB/s")

    t0 = time.time()
    for _ in range(nIter):
        geglu(input_data)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / nIter * 1000
    print(f"[GeGLU Perf]  {ms:.4f} ms | {gbps:.1f} GB/s")

    print("\n" + "=" * 60)
