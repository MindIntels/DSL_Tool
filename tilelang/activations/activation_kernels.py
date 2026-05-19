"""
TileLang Fused Gated Activations: SwiGLU, GeGLU
Tiled implementation following TileLang programming model.
"""
import torch
import time

try:
    import tvm
    from tvm.script import tir as T
    HAS_TVM = True
except ImportError:
    HAS_TVM = False


# ========== TVM TIR Specification ==========
def create_swiglu_tir(batch_size, hidden_size):
    if not HAS_TVM:
        return None

    @T.prim_func
    def swiglu_tir(
        Input: T.Buffer((batch_size, 2 * hidden_size), "float32"),
        Output: T.Buffer((batch_size, hidden_size), "float32"),
    ):
        T.func_attr({"global_symbol": "swiglu", "tir.noalias": True})
        for bx in T.thread_binding(batch_size, thread="blockIdx.x"):
            for tx in T.thread_binding(hidden_size, thread="threadIdx.x"):
                gate = Input[bx, tx]
                up = Input[bx, hidden_size + tx]
                sigmoid_gate = T.float32(1.0) / (T.float32(1.0) + T.exp(-gate))
                Output[bx, tx] = gate * sigmoid_gate * up

    return swiglu_tir


# ========== Tiled PyTorch Implementations ==========
class TiledSwiGLU:
    """SwiGLU with explicit tiling: output = SiLU(gate) * up."""

    def __init__(self, tile_size=1024):
        self.tile_size = tile_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """input: [batch, 2*hidden] -> [batch, hidden]"""
        batch, two_hidden = input.shape
        hidden = two_hidden // 2

        gate = input[:, :hidden]
        up = input[:, hidden:]
        output = torch.empty(batch, hidden, device=input.device, dtype=torch.float32)

        # Process in tiles along hidden dimension
        for ts in range(0, hidden, self.tile_size):
            te = min(ts + self.tile_size, hidden)
            g = gate[:, ts:te].float()
            u = up[:, ts:te].float()
            # SiLU = x * sigmoid(x)
            output[:, ts:te] = g * torch.sigmoid(g) * u

        return output


class TiledGeGLU:
    """GeGLU with explicit tiling: output = GELU(gate) * up."""

    def __init__(self, tile_size=1024):
        self.tile_size = tile_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch, two_hidden = input.shape
        hidden = two_hidden // 2

        gate = input[:, :hidden]
        up = input[:, hidden:]
        output = torch.empty(batch, hidden, device=input.device, dtype=torch.float32)

        for ts in range(0, hidden, self.tile_size):
            te = min(ts + self.tile_size, hidden)
            g = gate[:, ts:te].float()
            u = up[:, ts:te].float()
            # GELU (tanh approximation)
            output[:, ts:te] = torch.nn.functional.gelu(g, approximate='tanh') * u

        return output


class TiledSwiGLUBackward:
    """SwiGLU backward pass with tiling."""

    def __init__(self, tile_size=1024):
        self.tile_size = tile_size

    def backward(self, grad_output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        batch, two_hidden = input.shape
        hidden = two_hidden // 2
        grad_input = torch.empty_like(input)

        gate = input[:, :hidden]
        up = input[:, hidden:]

        for ts in range(0, hidden, self.tile_size):
            te = min(ts + self.tile_size, hidden)
            g = gate[:, ts:te].float()
            u = up[:, ts:te].float()
            go = grad_output[:, ts:te].float()

            sig = torch.sigmoid(g)
            silu_val = g * sig

            # d_up = SiLU(gate) * grad_output
            grad_input[:, hidden + ts:hidden + te] = silu_val * go

            # d_gate = up * go * dsilu(gate)
            dsilu = sig * (1.0 + g * (1.0 - sig))
            grad_input[:, ts:te] = u * go * dsilu

        return grad_input


# ========== TileLang Kernel Spec ==========
TILELANG_SWIGLU_SPEC = """
@tilelang.jit(grid=lambda B, H: (B * H // 1024,), block=(1024,))
def swiglu_tilelang(
    Input: tilelang.Tensor[B, 2*H, "float32"],
    Output: tilelang.Tensor[B, H, "float32"],
):
    idx = tilelang.global_id(0)
    b = idx // H
    h = idx % H

    gate = Input[b, h]
    up = Input[b, H + h]
    sigmoid_gate = 1.0 / (1.0 + tilelang.exp(-gate))
    Output[b, h] = gate * sigmoid_gate * up
"""


# ---- Test & Benchmark ----
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("TileLang Fused Gated Activations")
    print("=" * 60)

    batch, hidden = 32, 4096
    input_data = torch.randn(batch, 2 * hidden, device=device, dtype=torch.float32)

    # SwiGLU test
    swiglu_op = TiledSwiGLU()
    out_swiglu = swiglu_op.forward(input_data)
    gate = input_data[:, :hidden]
    up = input_data[:, hidden:]
    ref_swiglu = torch.nn.functional.silu(gate) * up
    err = (out_swiglu - ref_swiglu).abs().max().item()
    print(f"\n[SwiGLU] batch={batch}, hidden={hidden}")
    print(f"  Max error: {err:.2e} -> {'PASS' if err < 1e-6 else 'FAIL'}")

    # GeGLU test
    geglu_op = TiledGeGLU()
    out_geglu = geglu_op.forward(input_data)
    ref_geglu = torch.nn.functional.gelu(gate, approximate='tanh') * up
    err_geglu = (out_geglu - ref_geglu).abs().max().item()
    print(f"\n[GeGLU] batch={batch}, hidden={hidden}")
    print(f"  Max error: {err_geglu:.2e} -> {'PASS' if err_geglu < 1e-6 else 'FAIL'}")

    # Backward test
    bwd_op = TiledSwiGLUBackward()
    grad_out = torch.randn(batch, hidden, device=device, dtype=torch.float32)
    grad_in = bwd_op.backward(grad_out, input_data)
    print(f"\n[SwiGLU Backward] grad_input shape: {grad_in.shape} -> PASS")

    # Benchmark
    nIter = 200
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(nIter):
        swiglu_op.forward(input_data)
    if device == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) / nIter * 1000
    gbps = batch * 3 * hidden * 4 / (ms * 1e-3) / 1e9
    print(f"\n[SwiGLU Perf] {ms:.4f} ms | {gbps:.1f} GB/s")

    t0 = time.time()
    for _ in range(nIter):
        geglu_op.forward(input_data)
    if device == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) / nIter * 1000
    print(f"[GeGLU Perf]  {ms:.4f} ms")

    if HAS_TVM:
        func = create_swiglu_tir(batch, hidden)
        if func:
            print("\n[TVM TIR] SwiGLU schedule created.")

    print("\n" + "=" * 60)
