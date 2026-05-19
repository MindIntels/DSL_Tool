"""
Qwen3NeXt Kernels in Triton:
- Gated Softmax Attention
- Zero-Centered RMSNorm
- Gated Delta Rule / Gated DeltaNet
"""
import torch
import triton
import triton.language as tl
import time
import math


# ========== Zero-Centered RMSNorm ==========
@triton.jit
def zero_centered_rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_x, N,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """y = (1 + weight) * x / rms(x), weight initialized to 0"""
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


# ========== Gated Softmax Attention ==========
@triton.jit
def gated_softmax_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Gate_ptr, O_ptr,
    seq_len, head_dim,
    stride_qb, stride_qh, stride_qs, stride_qd,
    scale,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Gated softmax attention: O = softmax(Q@K^T/sqrt(d)) * gate @ V"""
    batch_idx = tl.program_id(2)
    head_idx = tl.program_id(1)
    query_idx = tl.program_id(0)

    # Load gate (sigmoid)
    g_raw = tl.load(Gate_ptr + head_idx)
    g = tl.sigmoid(g_raw)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    # Load query
    q_offset = batch_idx * stride_qb + head_idx * stride_qh + query_idx * stride_qs
    q = tl.load(Q_ptr + q_offset + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)

    # Compute attention scores and output in blocks over KV
    m_prev = float('-inf')
    l_prev = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for kv_start in range(0, seq_len, BLOCK_S):
        kv_offs = kv_start + tl.arange(0, BLOCK_S)
        kv_mask = (kv_offs < seq_len) & (kv_offs <= query_idx)  # causal mask

        # Compute QK^T for this block
        scores = tl.zeros([BLOCK_S], dtype=tl.float32)
        for d in range(0, head_dim):
            k_vals = tl.load(K_ptr + batch_idx * stride_qb + head_idx * stride_qh +
                           kv_offs * stride_qs + d * stride_qd,
                           mask=kv_mask, other=0.0).to(tl.float32)
            scores += q[d] * k_vals if d < head_dim else scores

        scores = scores * scale
        scores = tl.where(kv_mask, scores, float('-inf'))

        # Online softmax
        m_new = tl.maximum(m_prev, tl.max(scores, axis=0))
        exp_scores = tl.exp(scores - m_new)
        exp_scores = tl.where(kv_mask, exp_scores, 0.0)
        l_new = tl.exp(m_prev - m_new) * l_prev + tl.sum(exp_scores, axis=0)

        # Update accumulator
        acc = acc * tl.exp(m_prev - m_new)
        for s_idx in range(BLOCK_S):
            if kv_start + s_idx < seq_len and kv_start + s_idx <= query_idx:
                v_vals = tl.load(V_ptr + batch_idx * stride_qb + head_idx * stride_qh +
                               (kv_start + s_idx) * stride_qs + offs_d * stride_qd,
                               mask=d_mask, other=0.0).to(tl.float32)
                acc += exp_scores[s_idx] * v_vals

        m_prev = m_new
        l_prev = l_new

    # Normalize and apply gate
    acc = acc / (l_prev + 1e-6) * g

    o_offset = batch_idx * stride_qb + head_idx * stride_qh + query_idx * stride_qs
    tl.store(O_ptr + o_offset + offs_d * stride_qd, acc, mask=d_mask)


# ========== Gated Delta Rule (Sequential) ==========
@triton.jit
def gated_delta_rule_step_kernel(
    Q_ptr, K_ptr, V_ptr, Beta_ptr, S_ptr, O_ptr,
    head_dim, t,
    stride_b, stride_t, stride_d,
    BLOCK_D: tl.constexpr,
):
    """One step of gated delta rule. S_ptr is the state [batch, D, D]."""
    batch_idx = tl.program_id(0)
    d_out = tl.program_id(1)  # output row of state matrix

    offs_d = tl.arange(0, BLOCK_D)
    mask = offs_d < head_dim

    # Load Q, K, V for this timestep
    base_offset = batch_idx * stride_b + t * stride_t
    q = tl.load(Q_ptr + base_offset + offs_d * stride_d, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(K_ptr + base_offset + offs_d * stride_d, mask=mask, other=0.0).to(tl.float32)
    v_val = tl.load(V_ptr + base_offset + d_out * stride_d).to(tl.float32)
    beta_raw = tl.load(Beta_ptr + batch_idx * stride_b // head_dim + t)
    beta = tl.sigmoid(beta_raw)

    # Load S[d_out, :]
    s_offset = batch_idx * head_dim * head_dim + d_out * head_dim
    s_row = tl.load(S_ptr + s_offset + offs_d, mask=mask, other=0.0).to(tl.float32)

    # S @ k for this row
    sk = tl.sum(s_row * k, axis=0)

    # Update: S[d_out,:] += beta * (v[d_out] - sk) * k[:]
    s_row = s_row + beta * (v_val - sk) * k

    # Store updated state
    tl.store(S_ptr + s_offset + offs_d, s_row, mask=mask)

    # Output: o[d_out] = S[d_out, :] @ q
    o_val = tl.sum(s_row * q, axis=0)
    tl.store(O_ptr + base_offset + d_out * stride_d, o_val)


# ---- Python Wrappers ----
def zero_centered_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    batch = x.shape[0]
    N = x.shape[-1]
    y = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(N)
    zero_centered_rmsnorm_kernel[(batch,)](x, weight, y, x.stride(0), N, eps=eps, BLOCK_N=BLOCK_N)
    return y


def gated_softmax_attention(Q, K, V, gate, seq_len, head_dim):
    """Q,K,V: [batch, heads, seq, dim], gate: [heads]"""
    B, H, S, D = Q.shape
    O = torch.empty_like(Q)
    scale = 1.0 / math.sqrt(D)

    BLOCK_S = min(32, triton.next_power_of_2(S))
    BLOCK_D = triton.next_power_of_2(D)
    grid = (S, H, B)

    gated_softmax_attention_kernel[grid](
        Q, K, V, gate, O,
        S, D,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        scale,
        BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
    )
    return O


def gated_delta_rule(Q, K, V, beta, seq_len, head_dim):
    """
    Gated delta rule recurrence.
    Q,K,V: [batch, seq_len, head_dim], beta: [batch, seq_len]
    """
    B, S, D = Q.shape
    O = torch.zeros_like(Q)
    state = torch.zeros(B, D, D, device=Q.device, dtype=torch.float32)

    BLOCK_D = triton.next_power_of_2(D)

    for t in range(S):
        grid = (B, D)
        gated_delta_rule_step_kernel[grid](
            Q, K, V, beta, state, O,
            D, t,
            Q.stride(0), Q.stride(1), Q.stride(2),
            BLOCK_D=BLOCK_D,
        )
    return O


# ---- Test & Benchmark ----
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda"

    print("=" * 60)
    print("Triton Qwen3NeXt Kernels")
    print("=" * 60)

    # Zero-Centered RMSNorm
    batch, hidden = 32, 4096
    x = torch.randn(batch, hidden, device=device, dtype=torch.float32)
    w = torch.zeros(hidden, device=device, dtype=torch.float32)

    y = zero_centered_rmsnorm(x, w)
    rms = (x ** 2).mean(-1, keepdim=True).sqrt()
    ref = (1.0 + w) * x / (rms + 1e-6)
    err = (y - ref).abs().max().item()
    print(f"\n[Zero-Centered RMSNorm] batch={batch}, hidden={hidden}")
    print(f"  Max error: {err:.2e} -> {'PASS' if err < 1e-4 else 'FAIL'}")

    # Gated Delta Rule
    batch, seq, dim = 4, 64, 32
    Q = torch.randn(batch, seq, dim, device=device, dtype=torch.float32) * 0.1
    K = torch.randn(batch, seq, dim, device=device, dtype=torch.float32) * 0.1
    V = torch.randn(batch, seq, dim, device=device, dtype=torch.float32) * 0.1
    beta = torch.randn(batch, seq, device=device, dtype=torch.float32)

    O = gated_delta_rule(Q, K, V, beta, seq, dim)

    # CPU reference
    O_ref = torch.zeros_like(Q)
    for b in range(batch):
        S = torch.zeros(dim, dim, device=device)
        for t in range(seq):
            q = Q[b, t]
            k = K[b, t]
            v = V[b, t]
            bt = torch.sigmoid(beta[b, t])
            sk = S @ k
            S = S + bt * torch.outer(v - sk, k)  # simplified per-row
            O_ref[b, t] = S @ q

    # Note: The kernel processes per-row, matching reference
    err_dr = (O - O_ref).abs().max().item()
    print(f"\n[Gated Delta Rule] batch={batch}, seq={seq}, dim={dim}")
    print(f"  Max error: {err_dr:.2e} -> {'PASS' if err_dr < 1e-2 else 'FAIL'}")

    # Benchmark zero-centered rmsnorm
    nIter = 200
    x_bench = torch.randn(32, 4096, device=device, dtype=torch.float32)
    w_bench = torch.randn(4096, device=device, dtype=torch.float32) * 0.01
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(nIter):
        zero_centered_rmsnorm(x_bench, w_bench)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / nIter * 1000
    print(f"\n[Zero-Centered RMSNorm Perf] {ms:.4f} ms")

    print("\n" + "=" * 60)
