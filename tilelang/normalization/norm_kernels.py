"""
TileLang Normalization Kernels: RMSNorm, LayerNorm, Gemma-style
TVM TIR-based tile programming model.
"""
import torch
import numpy as np
import time

try:
    import tvm
    from tvm import te, tir
    from tvm.script import tir as T
    HAS_TVM = True
except ImportError:
    HAS_TVM = False


# ========== TVM TIR-based RMSNorm ==========
def create_rmsnorm_tir(batch_size, hidden_size, eps=1e-6):
    """Create TIR schedule for RMSNorm."""
    if not HAS_TVM:
        return None

    @T.prim_func
    def rmsnorm_tir(
        X: T.Buffer((batch_size, hidden_size), "float32"),
        W: T.Buffer((hidden_size,), "float32"),
        Y: T.Buffer((batch_size, hidden_size), "float32"),
    ):
        T.func_attr({"global_symbol": "rmsnorm", "tir.noalias": True})
        # Reduction buffer for sum of squares
        ss = T.alloc_buffer((batch_size,), "float32", scope="local")

        for b in T.thread_binding(batch_size, thread="blockIdx.x"):
            # Compute sum of squares
            ss[b] = T.float32(0)
            for i in T.serial(hidden_size):
                ss[b] = ss[b] + X[b, i] * X[b, i]

            # Normalize
            for i in T.thread_binding(hidden_size, thread="threadIdx.x"):
                rms_inv = T.rsqrt(ss[b] / hidden_size + eps)
                Y[b, i] = X[b, i] * rms_inv * W[i]

    return rmsnorm_tir


# ========== PyTorch Tiled Implementations (TileLang-style) ==========
class TiledRMSNorm:
    """RMSNorm with explicit tiling strategy (TileLang-style)."""

    def __init__(self, hidden_size, eps=1e-6, tile_size=256):
        self.hidden_size = hidden_size
        self.eps = eps
        self.tile_size = tile_size

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """x: [batch, hidden], weight: [hidden] -> y: [batch, hidden]"""
        batch = x.shape[0]
        hidden = x.shape[-1]

        # Phase 1: Compute RMS (tiled reduction)
        ss = torch.zeros(batch, device=x.device, dtype=torch.float32)
        for tile_start in range(0, hidden, self.tile_size):
            tile_end = min(tile_start + self.tile_size, hidden)
            x_tile = x[:, tile_start:tile_end].float()
            ss += (x_tile * x_tile).sum(dim=-1)

        rms_inv = torch.rsqrt(ss / hidden + self.eps).unsqueeze(-1)

        # Phase 2: Normalize with tiled multiply
        y = torch.empty_like(x, dtype=torch.float32)
        for tile_start in range(0, hidden, self.tile_size):
            tile_end = min(tile_start + self.tile_size, hidden)
            y[:, tile_start:tile_end] = (
                x[:, tile_start:tile_end].float() * rms_inv *
                weight[tile_start:tile_end].float()
            )
        return y


class TiledLayerNorm:
    """LayerNorm with explicit tiling."""

    def __init__(self, hidden_size, eps=1e-5, tile_size=256):
        self.hidden_size = hidden_size
        self.eps = eps
        self.tile_size = tile_size

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        hidden = x.shape[-1]

        # Phase 1: Compute mean (tiled)
        total = torch.zeros(batch, device=x.device, dtype=torch.float32)
        for ts in range(0, hidden, self.tile_size):
            te = min(ts + self.tile_size, hidden)
            total += x[:, ts:te].float().sum(dim=-1)
        mean = (total / hidden).unsqueeze(-1)

        # Phase 2: Compute variance (tiled)
        var_sum = torch.zeros(batch, device=x.device, dtype=torch.float32)
        for ts in range(0, hidden, self.tile_size):
            te_idx = min(ts + self.tile_size, hidden)
            diff = x[:, ts:te_idx].float() - mean
            var_sum += (diff * diff).sum(dim=-1)
        rstd = torch.rsqrt(var_sum / hidden + self.eps).unsqueeze(-1)

        # Phase 3: Normalize (tiled)
        y = torch.empty_like(x, dtype=torch.float32)
        for ts in range(0, hidden, self.tile_size):
            te_idx = min(ts + self.tile_size, hidden)
            normalized = (x[:, ts:te_idx].float() - mean) * rstd
            y[:, ts:te_idx] = normalized * weight[ts:te_idx].float() + bias[ts:te_idx].float()
        return y


class TiledGemmaRMSNorm:
    """Gemma-style RMSNorm: y = (1 + w) * x / rms(x)."""

    def __init__(self, hidden_size, eps=1e-6, tile_size=256):
        self.hidden_size = hidden_size
        self.eps = eps
        self.tile_size = tile_size

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        hidden = x.shape[-1]

        # Phase 1: RMS
        ss = torch.zeros(batch, device=x.device, dtype=torch.float32)
        for ts in range(0, hidden, self.tile_size):
            te_idx = min(ts + self.tile_size, hidden)
            x_tile = x[:, ts:te_idx].float()
            ss += (x_tile * x_tile).sum(dim=-1)
        rms_inv = torch.rsqrt(ss / hidden + self.eps).unsqueeze(-1)

        # Phase 2: (1 + w) * x * rms_inv
        y = torch.empty_like(x, dtype=torch.float32)
        for ts in range(0, hidden, self.tile_size):
            te_idx = min(ts + self.tile_size, hidden)
            y[:, ts:te_idx] = (
                (1.0 + weight[ts:te_idx].float()) *
                x[:, ts:te_idx].float() * rms_inv
            )
        return y


# ========== TileLang Kernel Spec ==========
TILELANG_RMSNORM_SPEC = """
# TileLang RMSNorm Kernel Specification

@tilelang.jit(grid=lambda batch: (batch,), block=(1024,))
def rmsnorm_tilelang(
    X: tilelang.Tensor[batch, hidden, "float32"],
    W: tilelang.Tensor[hidden, "float32"],
    Y: tilelang.Tensor[batch, hidden, "float32"],
    eps: float = 1e-6,
):
    row = tilelang.block_id(0)

    # Cooperative reduction for sum of squares
    ss = tilelang.reduce_sum(X[row, :] * X[row, :])
    rms_inv = tilelang.rsqrt(ss / hidden + eps)

    # Vectorized normalize + scale
    Y[row, :] = X[row, :] * rms_inv * W[:]
"""


# ---- Test & Benchmark ----
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("TileLang Normalization Kernels (Tiled)")
    print("=" * 60)

    batch, hidden = 32, 4096

    x = torch.randn(batch, hidden, device=device, dtype=torch.float32)
    w = torch.ones(hidden, device=device, dtype=torch.float32)
    b = torch.zeros(hidden, device=device, dtype=torch.float32)

    # RMSNorm
    rms_op = TiledRMSNorm(hidden)
    y = rms_op.forward(x, w)
    rms = (x ** 2).mean(-1, keepdim=True).sqrt()
    ref = x / (rms + 1e-6)
    err = (y - ref).abs().max().item()
    print(f"\n[RMSNorm] batch={batch}, hidden={hidden}")
    print(f"  Max error: {err:.2e} -> {'PASS' if err < 1e-5 else 'FAIL'}")

    # LayerNorm
    ln_op = TiledLayerNorm(hidden)
    y_ln = ln_op.forward(x, w, b)
    ref_ln = torch.nn.functional.layer_norm(x, [hidden], w, b)
    err_ln = (y_ln - ref_ln).abs().max().item()
    print(f"\n[LayerNorm] batch={batch}, hidden={hidden}")
    print(f"  Max error: {err_ln:.2e} -> {'PASS' if err_ln < 1e-5 else 'FAIL'}")

    # Gemma RMSNorm
    gemma_op = TiledGemmaRMSNorm(hidden)
    w_zero = torch.zeros(hidden, device=device, dtype=torch.float32)
    y_g = gemma_op.forward(x, w_zero)
    ref_g = x / (rms + 1e-6)
    err_g = (y_g - ref_g).abs().max().item()
    print(f"\n[Gemma RMSNorm] batch={batch}, hidden={hidden}")
    print(f"  Max error: {err_g:.2e} -> {'PASS' if err_g < 1e-5 else 'FAIL'}")

    # Benchmark
    nIter = 100
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(nIter):
        rms_op.forward(x, w)
    if device == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) / nIter * 1000
    gbps = 2 * batch * hidden * 4 / (ms * 1e-3) / 1e9
    print(f"\n[RMSNorm Perf] {ms:.4f} ms | {gbps:.1f} GB/s")

    if HAS_TVM:
        print("\n[TVM TIR] RMSNorm schedule created successfully")
        func = create_rmsnorm_tir(batch, hidden)
        if func:
            print("  TIR function generated.")

    print("\n" + "=" * 60)
