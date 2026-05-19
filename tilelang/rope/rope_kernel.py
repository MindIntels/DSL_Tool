"""
TileLang RoPE: Rotary Position Embeddings (LLaMA style + LLaMA 3.1)
"""
import torch
import math
import time

try:
    import tvm
    from tvm.script import tir as T
    HAS_TVM = True
except ImportError:
    HAS_TVM = False


class TiledRoPE:
    """Standard RoPE with tiled application."""

    def __init__(self, head_dim, base=10000.0, tile_size=64):
        self.head_dim = head_dim
        self.base = base
        self.tile_size = tile_size
        self.half_dim = head_dim // 2

    def forward(self, x: torch.Tensor, positions=None) -> torch.Tensor:
        """x: [batch, seq_len, num_heads, head_dim]"""
        B, S, H, D = x.shape
        half = D // 2
        device = x.device

        # Compute frequencies
        freqs = 1.0 / (self.base ** (torch.arange(0, half, device=device).float() * 2 / D))

        if positions is None:
            positions = torch.arange(S, device=device).float()
        else:
            positions = positions.float()

        # Apply in tiles over sequence length
        out = torch.empty_like(x)
        for s_start in range(0, S, self.tile_size):
            s_end = min(s_start + self.tile_size, S)
            pos_tile = positions[s_start:s_end]

            # [tile_size, half_dim]
            theta = torch.outer(pos_tile, freqs)
            cos_t = theta.cos()  # [tile, half]
            sin_t = theta.sin()

            # Reshape for broadcasting: [1, tile, 1, half]
            cos_t = cos_t[None, :, None, :]
            sin_t = sin_t[None, :, None, :]

            x0 = x[:, s_start:s_end, :, :half]
            x1 = x[:, s_start:s_end, :, half:]

            out[:, s_start:s_end, :, :half] = x0 * cos_t - x1 * sin_t
            out[:, s_start:s_end, :, half:] = x0 * sin_t + x1 * cos_t

        return out


class TiledRoPELLaMA31:
    """LLaMA 3.1 RoPE with frequency-based scaling for extended context."""

    def __init__(self, head_dim, base=500000.0, scaling_factor=8.0,
                 low_freq_factor=1.0, high_freq_factor=4.0,
                 original_max_pos=8192, tile_size=64):
        self.head_dim = head_dim
        self.base = base
        self.scaling_factor = scaling_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.original_max_pos = original_max_pos
        self.tile_size = tile_size

    def _compute_frequencies(self, device):
        half = self.head_dim // 2
        base_freqs = 1.0 / (self.base ** (torch.arange(0, half, device=device).float() * 2 / self.head_dim))
        wavelengths = 2 * math.pi / base_freqs

        old_ctx = float(self.original_max_pos)
        low_bound = self.low_freq_factor * old_ctx
        high_bound = self.high_freq_factor * old_ctx

        freqs = torch.empty_like(base_freqs)
        for i in range(half):
            wl = wavelengths[i].item()
            if wl < low_bound:
                # High frequency: scale
                freqs[i] = base_freqs[i] / self.scaling_factor
            elif wl > high_bound:
                # Low frequency: keep
                freqs[i] = base_freqs[i]
            else:
                # Transition: smooth interpolation
                smooth = (old_ctx / wl - self.low_freq_factor) / (self.high_freq_factor - self.low_freq_factor)
                freqs[i] = (1.0 - smooth) * base_freqs[i] / self.scaling_factor + smooth * base_freqs[i]

        return freqs

    def forward(self, x: torch.Tensor, positions=None) -> torch.Tensor:
        B, S, H, D = x.shape
        half = D // 2
        device = x.device

        freqs = self._compute_frequencies(device)
        if positions is None:
            positions = torch.arange(S, device=device).float()

        out = torch.empty_like(x)
        for s_start in range(0, S, self.tile_size):
            s_end = min(s_start + self.tile_size, S)
            pos_tile = positions[s_start:s_end]

            theta = torch.outer(pos_tile, freqs)
            cos_t = theta.cos()[None, :, None, :]
            sin_t = theta.sin()[None, :, None, :]

            x0 = x[:, s_start:s_end, :, :half]
            x1 = x[:, s_start:s_end, :, half:]

            out[:, s_start:s_end, :, :half] = x0 * cos_t - x1 * sin_t
            out[:, s_start:s_end, :, half:] = x0 * sin_t + x1 * cos_t

        return out


# ========== TVM TIR Spec ==========
def create_rope_tir(batch, seq_len, num_heads, head_dim, base=10000.0):
    if not HAS_TVM:
        return None
    half_dim = head_dim // 2

    @T.prim_func
    def rope_tir(
        X: T.Buffer((batch, seq_len, num_heads, head_dim), "float32"),
        Y: T.Buffer((batch, seq_len, num_heads, head_dim), "float32"),
    ):
        T.func_attr({"global_symbol": "rope", "tir.noalias": True})
        for b in T.thread_binding(batch, thread="blockIdx.z"):
            for s in T.thread_binding(seq_len, thread="blockIdx.y"):
                for h in T.thread_binding(num_heads, thread="blockIdx.x"):
                    for d in T.thread_binding(half_dim, thread="threadIdx.x"):
                        freq = T.float32(1.0) / T.pow(T.float32(base), T.float32(2 * d) / T.float32(head_dim))
                        theta = T.cast(s, "float32") * freq
                        cos_t = T.cos(theta)
                        sin_t = T.sin(theta)
                        x0 = X[b, s, h, d]
                        x1 = X[b, s, h, d + half_dim]
                        Y[b, s, h, d] = x0 * cos_t - x1 * sin_t
                        Y[b, s, h, d + half_dim] = x0 * sin_t + x1 * cos_t

    return rope_tir


# ---- Test & Benchmark ----
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("TileLang RoPE Kernels")
    print("=" * 60)

    batch, seq_len, num_heads, head_dim = 2, 128, 32, 128
    x = torch.randn(batch, seq_len, num_heads, head_dim, device=device, dtype=torch.float32)

    # Standard RoPE
    rope = TiledRoPE(head_dim)
    out = rope.forward(x)

    # Reference
    half = head_dim // 2
    freqs = 1.0 / (10000.0 ** (torch.arange(0, half, device=device).float() * 2 / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    theta = torch.outer(positions, freqs)
    cos_t = theta.cos()[None, :, None, :]
    sin_t = theta.sin()[None, :, None, :]
    x0, x1 = x[..., :half], x[..., half:]
    ref = torch.cat([x0 * cos_t - x1 * sin_t, x0 * sin_t + x1 * cos_t], dim=-1)

    err = (out - ref).abs().max().item()
    print(f"\n[Standard RoPE] batch={batch}, seq={seq_len}, heads={num_heads}, dim={head_dim}")
    print(f"  Max error: {err:.2e} -> {'PASS' if err < 1e-6 else 'FAIL'}")

    # LLaMA 3.1 RoPE
    rope31 = TiledRoPELLaMA31(head_dim)
    out31 = rope31.forward(x)
    valid = not (torch.isnan(out31).any() or torch.isinf(out31).any())
    print(f"\n[LLaMA 3.1 RoPE] Output valid: {'PASS' if valid else 'FAIL'}")

    # Benchmark
    nIter = 100
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(nIter):
        rope.forward(x)
    if device == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) / nIter * 1000
    gbps = 2 * batch * seq_len * num_heads * head_dim * 4 / (ms * 1e-3) / 1e9
    print(f"\n[Standard RoPE Perf] {ms:.4f} ms | {gbps:.1f} GB/s")

    t0 = time.time()
    for _ in range(nIter):
        rope31.forward(x)
    if device == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) / nIter * 1000
    print(f"[LLaMA 3.1 RoPE Perf] {ms:.4f} ms")

    print("\n" + "=" * 60)
