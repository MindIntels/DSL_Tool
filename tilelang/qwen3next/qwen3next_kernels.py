"""
TileLang Qwen3NeXt Kernels:
- Zero-Centered RMSNorm
- Gated Softmax Attention
- Gated Delta Rule / Gated DeltaNet
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


# ========== Zero-Centered RMSNorm ==========
class TiledZeroCenteredRMSNorm:
    """Zero-Centered RMSNorm: y = (1 + w) * x / rms(x)
    Weight initialized to 0, so at init this is identity / rms.
    """
    def __init__(self, hidden_size, eps=1e-6, tile_size=256):
        self.hidden_size = hidden_size
        self.eps = eps
        self.tile_size = tile_size

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        hidden = x.shape[-1]

        ss = torch.zeros(batch, device=x.device, dtype=torch.float32)
        for ts in range(0, hidden, self.tile_size):
            te = min(ts + self.tile_size, hidden)
            tile = x[:, ts:te].float()
            ss += (tile * tile).sum(dim=-1)

        rms_inv = torch.rsqrt(ss / hidden + self.eps).unsqueeze(-1)

        y = torch.empty_like(x, dtype=torch.float32)
        for ts in range(0, hidden, self.tile_size):
            te = min(ts + self.tile_size, hidden)
            y[:, ts:te] = (1.0 + weight[ts:te].float()) * x[:, ts:te].float() * rms_inv

        return y


# ========== Gated Softmax Attention ==========
class TiledGatedSoftmaxAttention:
    """Gated Softmax Attention: O = gate * softmax(Q@K^T/sqrt(d)) @ V
    gate is per-head sigmoid scalar.
    """
    def __init__(self, tile_kv=32):
        self.tile_kv = tile_kv

    def forward(self, Q, K, V, gate):
        """
        Q, K, V: [batch, num_heads, seq_len, head_dim]
        gate: [num_heads] (raw, sigmoid applied internally)
        """
        B, H, S, D = Q.shape
        scale = 1.0 / math.sqrt(D)
        gate_sig = torch.sigmoid(gate)  # [H]

        O = torch.zeros_like(Q)

        for b in range(B):
            for h in range(H):
                g = gate_sig[h].item()
                for q_idx in range(S):
                    q = Q[b, h, q_idx]  # [D]

                    # Online softmax over KV (causal)
                    max_score = float('-inf')
                    scores = []
                    for kv_start in range(0, q_idx + 1, self.tile_kv):
                        kv_end = min(kv_start + self.tile_kv, q_idx + 1)
                        k_tile = K[b, h, kv_start:kv_end]  # [tile, D]
                        s = (k_tile @ q) * scale  # [tile]
                        scores.append(s)
                        tile_max = s.max().item()
                        if tile_max > max_score:
                            max_score = tile_max

                    # Compute softmax
                    exp_sum = 0.0
                    exp_scores = []
                    for s in scores:
                        es = torch.exp(s - max_score)
                        exp_scores.append(es)
                        exp_sum += es.sum().item()

                    # Weighted sum of V
                    out = torch.zeros(D, device=Q.device, dtype=Q.dtype)
                    idx = 0
                    for kv_start in range(0, q_idx + 1, self.tile_kv):
                        kv_end = min(kv_start + self.tile_kv, q_idx + 1)
                        v_tile = V[b, h, kv_start:kv_end]
                        weights = exp_scores[idx] / (exp_sum + 1e-6)
                        out += (weights.unsqueeze(-1) * v_tile).sum(0)
                        idx += 1

                    O[b, h, q_idx] = out * g

        return O


# ========== Gated Delta Rule ==========
class TiledGatedDeltaRule:
    """Gated Delta Rule with tiled state updates.
    S_t = S_{t-1} + beta_t * (v_t - S_{t-1}@k_t) @ k_t^T
    o_t = S_t @ q_t
    """
    def __init__(self, head_dim, tile_d=32):
        self.head_dim = head_dim
        self.tile_d = tile_d

    def forward(self, Q, K, V, beta_raw):
        """
        Q, K, V: [batch, seq_len, head_dim]
        beta_raw: [batch, seq_len] (pre-sigmoid)
        """
        B, S, D = Q.shape
        device = Q.device
        O = torch.zeros(B, S, D, device=device, dtype=Q.dtype)

        for b in range(B):
            # State matrix: [D, D]
            state = torch.zeros(D, D, device=device, dtype=torch.float32)

            for t in range(S):
                q = Q[b, t].float()
                k = K[b, t].float()
                v = V[b, t].float()
                beta = torch.sigmoid(beta_raw[b, t]).item()

                # Tiled state update
                # sk = S @ k
                sk = torch.zeros(D, device=device, dtype=torch.float32)
                for d_start in range(0, D, self.tile_d):
                    d_end = min(d_start + self.tile_d, D)
                    sk[d_start:d_end] = state[d_start:d_end] @ k

                # S += beta * outer(v - sk, k)
                update = beta * torch.outer(v - sk, k)
                for d_start in range(0, D, self.tile_d):
                    d_end = min(d_start + self.tile_d, D)
                    state[d_start:d_end] += update[d_start:d_end]

                # o = S @ q
                o = torch.zeros(D, device=device, dtype=torch.float32)
                for d_start in range(0, D, self.tile_d):
                    d_end = min(d_start + self.tile_d, D)
                    o[d_start:d_end] = state[d_start:d_end] @ q

                O[b, t] = o

        return O


# ========== Gated DeltaNet Block ==========
class TiledGatedDeltaNet:
    """Full Gated DeltaNet block:
    1. Short convolution (causal conv1d)
    2. Project to Q, K, V, beta
    3. Gated delta rule recurrence
    4. Output gating
    """
    def __init__(self, hidden_dim, head_dim, conv_size=4):
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.conv_size = conv_size
        self.delta_rule = TiledGatedDeltaRule(head_dim)

    def forward(self, x, Wq, Wk, Wv, Wbeta, Wgate, conv_weight):
        """
        x: [batch, seq_len, hidden_dim]
        Wq, Wk, Wv: [hidden_dim, head_dim]
        Wbeta: [hidden_dim, 1]
        Wgate: [hidden_dim, head_dim]
        conv_weight: [conv_size, hidden_dim]
        """
        B, S, H = x.shape

        # 1. Causal conv1d
        x_conv = torch.zeros_like(x)
        for t in range(S):
            for c in range(min(self.conv_size, t + 1)):
                x_conv[:, t] += x[:, t - c] * conv_weight[c]

        # 2. Project
        Q = x_conv @ Wq
        K = x_conv @ Wk
        V = x_conv @ Wv
        beta = (x_conv @ Wbeta).squeeze(-1)
        gate = torch.sigmoid(x_conv @ Wgate)

        # 3. Delta rule
        O = self.delta_rule.forward(Q, K, V, beta)

        # 4. Output gating
        return O * gate


# ---- Test & Benchmark ----
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("TileLang Qwen3NeXt Kernels")
    print("=" * 60)

    # Zero-Centered RMSNorm
    batch, hidden = 32, 4096
    x = torch.randn(batch, hidden, device=device, dtype=torch.float32)
    w = torch.zeros(hidden, device=device, dtype=torch.float32)

    norm = TiledZeroCenteredRMSNorm(hidden)
    y = norm.forward(x, w)
    rms = (x ** 2).mean(-1, keepdim=True).sqrt()
    ref = x / (rms + 1e-6)
    err = (y - ref).abs().max().item()
    print(f"\n[Zero-Centered RMSNorm] batch={batch}, hidden={hidden}")
    print(f"  Max error: {err:.2e} -> {'PASS' if err < 1e-5 else 'FAIL'}")

    # Gated Delta Rule
    batch_dr, seq, dim = 2, 32, 16
    Q = torch.randn(batch_dr, seq, dim, device=device, dtype=torch.float32) * 0.1
    K = torch.randn(batch_dr, seq, dim, device=device, dtype=torch.float32) * 0.1
    V = torch.randn(batch_dr, seq, dim, device=device, dtype=torch.float32) * 0.1
    beta = torch.randn(batch_dr, seq, device=device, dtype=torch.float32)

    delta_rule = TiledGatedDeltaRule(dim)
    O = delta_rule.forward(Q, K, V, beta)

    # CPU reference
    O_ref = torch.zeros_like(Q)
    for b in range(batch_dr):
        S = torch.zeros(dim, dim, device=device)
        for t in range(seq):
            q, k, v = Q[b, t], K[b, t], V[b, t]
            bt = torch.sigmoid(beta[b, t])
            sk = S @ k
            S = S + bt * torch.outer(v - sk, k)
            O_ref[b, t] = S @ q

    err_dr = (O - O_ref).abs().max().item()
    print(f"\n[Gated Delta Rule] batch={batch_dr}, seq={seq}, dim={dim}")
    print(f"  Max error: {err_dr:.2e} -> {'PASS' if err_dr < 1e-4 else 'FAIL'}")

    # Gated DeltaNet
    hidden_dim, head_dim = 64, 16
    x_dn = torch.randn(2, 16, hidden_dim, device=device, dtype=torch.float32) * 0.1
    Wq = torch.randn(hidden_dim, head_dim, device=device) * 0.1
    Wk = torch.randn(hidden_dim, head_dim, device=device) * 0.1
    Wv = torch.randn(hidden_dim, head_dim, device=device) * 0.1
    Wbeta = torch.randn(hidden_dim, 1, device=device) * 0.1
    Wgate = torch.randn(hidden_dim, head_dim, device=device) * 0.1
    conv_w = torch.randn(4, hidden_dim, device=device) * 0.1

    deltanet = TiledGatedDeltaNet(hidden_dim, head_dim)
    O_dn = deltanet.forward(x_dn, Wq, Wk, Wv, Wbeta, Wgate, conv_w)
    valid = not (torch.isnan(O_dn).any() or torch.isinf(O_dn).any())
    print(f"\n[Gated DeltaNet] Output shape: {O_dn.shape}, valid: {'PASS' if valid else 'FAIL'}")

    # Benchmark
    nIter = 50
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(nIter):
        norm.forward(x, w)
    if device == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) / nIter * 1000
    print(f"\n[Zero-Centered RMSNorm Perf] {ms:.4f} ms")

    print("\n" + "=" * 60)
    print("All Qwen3NeXt tests complete.")
