/**
 * Gated Delta Rule & Gated DeltaNet (Qwen3NeXt)
 *
 * Delta Rule: A linear attention variant that maintains a memory matrix S
 *   S_t = S_{t-1} + v_t @ k_t^T - (S_{t-1} @ k_t) @ k_t^T  (retrieval-and-update)
 *   o_t = S_t @ q_t
 *
 * Gated Delta Rule: Adds a learnable gate to control the update:
 *   S_t = (1 - beta_t * k_t @ k_t^T) * S_{t-1} + beta_t * v_t @ k_t^T
 *   o_t = S_t @ q_t
 *
 * Gated DeltaNet: Full architecture combining gated delta rule with:
 *   - Input/output gating
 *   - Short convolution (causal conv1d)
 *   - Gated output projection
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ========== Gated Delta Rule Recurrence ==========
// Processes one head at a time, sequentially over time steps
// Q, K, V: [batch, seq_len, head_dim]
// beta: [batch, seq_len, 1] - per-step gate (sigmoid applied inside)
// output: [batch, seq_len, head_dim]
__global__ void gated_delta_rule_kernel(
    const float *__restrict__ Q,
    const float *__restrict__ K,
    const float *__restrict__ V,
    const float *__restrict__ beta_raw,  // pre-sigmoid
    float *__restrict__ output,
    int seq_len, int head_dim) {

    int batch_idx = blockIdx.x;
    int d_out = threadIdx.x;  // output dimension
    if (d_out >= head_dim) return;

    // Each thread maintains one row of the state matrix S[d_out, :]
    // S: [head_dim, head_dim] - but we only need row d_out
    extern __shared__ float smem[];
    float *S_row = smem + threadIdx.x * head_dim;  // S[d_out, :]

    // Initialize state to zero
    for (int d = 0; d < head_dim; d++) S_row[d] = 0.0f;

    size_t batch_off = (size_t)batch_idx * seq_len * head_dim;

    for (int t = 0; t < seq_len; t++) {
        const float *q_t = Q + batch_off + t * head_dim;
        const float *k_t = K + batch_off + t * head_dim;
        const float *v_t = V + batch_off + t * head_dim;
        float beta = 1.0f / (1.0f + expf(-beta_raw[batch_idx * seq_len + t]));

        // Compute S @ k_t for this row (retrieval)
        float sk = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            sk += S_row[d] * k_t[d];
        }

        // Update: S_t[d_out, :] = S_{t-1}[d_out, :] + beta * (v_t[d_out] - sk) * k_t[:]
        float v_val = v_t[d_out];
        for (int d = 0; d < head_dim; d++) {
            S_row[d] += beta * (v_val - sk) * k_t[d];
        }

        // Output: o_t[d_out] = S_t[d_out, :] @ q_t
        float o = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            o += S_row[d] * q_t[d];
        }
        output[batch_off + t * head_dim + d_out] = o;
    }
}

// ========== Gated DeltaNet Block ==========
// Full block: conv1d -> gated delta rule -> gated output
// input: [batch, seq_len, hidden_dim]
// Simplified version focusing on the key computation
__global__ void gated_deltanet_kernel(
    const float *__restrict__ input,
    const float *__restrict__ Wq,    // [hidden_dim, head_dim]
    const float *__restrict__ Wk,    // [hidden_dim, head_dim]
    const float *__restrict__ Wv,    // [hidden_dim, head_dim]
    const float *__restrict__ Wbeta, // [hidden_dim, 1]
    const float *__restrict__ Wg,    // [hidden_dim, head_dim] (output gate)
    const float *__restrict__ Wo,    // [head_dim, hidden_dim]
    float *__restrict__ output,
    int seq_len, int hidden_dim, int head_dim) {

    int batch_idx = blockIdx.x;
    int d_out = threadIdx.x;
    if (d_out >= head_dim) return;

    extern __shared__ float smem[];
    float *S_row = smem + threadIdx.x * head_dim;
    float *q_buf = smem + blockDim.x * head_dim;
    float *k_buf = q_buf + head_dim;
    float *v_buf = k_buf + head_dim;

    for (int d = 0; d < head_dim; d++) S_row[d] = 0.0f;

    size_t batch_off_in = (size_t)batch_idx * seq_len * hidden_dim;
    size_t batch_off_hd = (size_t)batch_idx * seq_len * head_dim;

    for (int t = 0; t < seq_len; t++) {
        const float *x_t = input + batch_off_in + t * hidden_dim;

        // Project to Q, K, V (simplified - full version would use shared mem)
        float q_val = 0, k_val = 0, v_val = 0;
        for (int h = 0; h < hidden_dim; h++) {
            q_val += x_t[h] * Wq[h * head_dim + d_out];
            k_val += x_t[h] * Wk[h * head_dim + d_out];
            v_val += x_t[h] * Wv[h * head_dim + d_out];
        }

        // Compute beta
        float beta_raw = 0;
        for (int h = 0; h < hidden_dim; h++) beta_raw += x_t[h] * Wbeta[h];
        float beta = 1.0f / (1.0f + expf(-beta_raw));

        // Normalize K (L2 norm for stability)
        // In full version, this would be a block-level reduction
        // Here simplified per-thread

        // Delta rule update
        float sk = 0.0f;
        for (int d = 0; d < head_dim; d++) sk += S_row[d] * k_val;  // simplified

        for (int d = 0; d < head_dim; d++) {
            S_row[d] += beta * (v_val - sk) * k_val;  // simplified
        }

        // Output with gating
        float o = 0.0f;
        for (int d = 0; d < head_dim; d++) o += S_row[d] * q_val;

        // Output gate
        float gate = 0;
        for (int h = 0; h < hidden_dim; h++) gate += x_t[h] * Wg[h * head_dim + d_out];
        gate = 1.0f / (1.0f + expf(-gate));  // sigmoid gate

        output[batch_off_hd + t * head_dim + d_out] = o * gate;
    }
}

// ========== Chunked Parallel Delta Rule (for training) ==========
// Processes chunks in parallel using matrix operations within each chunk,
// then propagates state between chunks sequentially
__global__ void chunked_delta_rule_kernel(
    const float *__restrict__ Q,   // [batch, seq_len, head_dim]
    const float *__restrict__ K,
    const float *__restrict__ V,
    const float *__restrict__ beta_raw,
    float *__restrict__ output,
    int seq_len, int head_dim, int chunk_size) {

    int batch_idx = blockIdx.x;
    int chunk_idx = blockIdx.y;
    int d_out = threadIdx.x;
    if (d_out >= head_dim) return;

    extern __shared__ float smem[];
    float *S_row = smem + threadIdx.x * head_dim;
    for (int d = 0; d < head_dim; d++) S_row[d] = 0.0f;

    size_t batch_off = (size_t)batch_idx * seq_len * head_dim;
    int t_start = chunk_idx * chunk_size;
    int t_end = min(t_start + chunk_size, seq_len);

    // TODO: In full implementation, inter-chunk state would be passed via global memory
    // For now, process each chunk with local state
    for (int t = t_start; t < t_end; t++) {
        const float *q_t = Q + batch_off + t * head_dim;
        const float *k_t = K + batch_off + t * head_dim;
        const float *v_t = V + batch_off + t * head_dim;
        float beta = 1.0f / (1.0f + expf(-beta_raw[batch_idx * seq_len + t]));

        float sk = 0.0f;
        for (int d = 0; d < head_dim; d++) sk += S_row[d] * k_t[d];

        float v_val = v_t[d_out];
        for (int d = 0; d < head_dim; d++) {
            S_row[d] += beta * (v_val - sk) * k_t[d];
        }

        float o = 0.0f;
        for (int d = 0; d < head_dim; d++) o += S_row[d] * q_t[d];
        output[batch_off + t * head_dim + d_out] = o;
    }
}

// Host wrappers
void gated_delta_rule(const float *Q, const float *K, const float *V,
                       const float *beta, float *output,
                       int batch, int seq_len, int head_dim) {
    int threads = head_dim;
    int smem = head_dim * head_dim * sizeof(float);
    gated_delta_rule_kernel<<<batch, threads, smem>>>(
        Q, K, V, beta, output, seq_len, head_dim);
}

// ---- Test ----
int main() {
    int batch = 4, seq = 128, dim = 64;
    printf("Gated Delta Rule & DeltaNet: batch=%d, seq=%d, dim=%d\n", batch, seq, dim);

    size_t qkvSize = (size_t)batch * seq * dim * sizeof(float);
    size_t betaSize = (size_t)batch * seq * sizeof(float);

    float *hQ = (float *)malloc(qkvSize);
    float *hK = (float *)malloc(qkvSize);
    float *hV = (float *)malloc(qkvSize);
    float *hBeta = (float *)malloc(betaSize);
    float *hO = (float *)malloc(qkvSize);
    float *hO_ref = (float *)malloc(qkvSize);

    srand(42);
    for (size_t i = 0; i < (size_t)batch * seq * dim; i++) {
        hQ[i] = ((float)(rand() % 200) - 100) / 1000.0f;
        hK[i] = ((float)(rand() % 200) - 100) / 1000.0f;
        hV[i] = ((float)(rand() % 200) - 100) / 1000.0f;
    }
    for (size_t i = 0; i < (size_t)batch * seq; i++)
        hBeta[i] = ((float)(rand() % 200) - 100) / 100.0f;

    // CPU reference for gated delta rule
    for (int b = 0; b < batch; b++) {
        // State matrix S[dim, dim]
        float *S = (float *)calloc(dim * dim, sizeof(float));
        for (int t = 0; t < seq; t++) {
            float beta = 1.0f / (1.0f + expf(-hBeta[b * seq + t]));
            const float *q = hQ + (b * seq + t) * dim;
            const float *k = hK + (b * seq + t) * dim;
            const float *v = hV + (b * seq + t) * dim;

            for (int i = 0; i < dim; i++) {
                // S @ k
                float sk = 0;
                for (int j = 0; j < dim; j++) sk += S[i * dim + j] * k[j];
                // Update S
                for (int j = 0; j < dim; j++) {
                    S[i * dim + j] += beta * (v[i] - sk) * k[j];
                }
            }
            // Output: o = S @ q
            for (int i = 0; i < dim; i++) {
                float o = 0;
                for (int j = 0; j < dim; j++) o += S[i * dim + j] * q[j];
                hO_ref[(b * seq + t) * dim + i] = o;
            }
        }
        free(S);
    }

    float *dQ, *dK, *dV, *dBeta, *dO;
    CHECK_CUDA(cudaMalloc(&dQ, qkvSize));
    CHECK_CUDA(cudaMalloc(&dK, qkvSize));
    CHECK_CUDA(cudaMalloc(&dV, qkvSize));
    CHECK_CUDA(cudaMalloc(&dBeta, betaSize));
    CHECK_CUDA(cudaMalloc(&dO, qkvSize));
    CHECK_CUDA(cudaMemcpy(dQ, hQ, qkvSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, hK, qkvSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, hV, qkvSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dBeta, hBeta, betaSize, cudaMemcpyHostToDevice));

    gated_delta_rule(dQ, dK, dV, dBeta, dO, batch, seq, dim);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hO, dO, qkvSize, cudaMemcpyDeviceToHost));

    float maxErr = 0;
    for (size_t i = 0; i < (size_t)batch * seq * dim; i++) {
        float err = fabsf(hO[i] - hO_ref[i]);
        if (err > maxErr) maxErr = err;
    }
    printf("[Gated Delta Rule] Max error: %e -> %s\n", maxErr, maxErr < 1e-2f ? "PASS" : "FAIL");

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    int nIter = 50;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < nIter; i++)
        gated_delta_rule(dQ, dK, dV, dBeta, dO, batch, seq, dim);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= nIter;
    printf("Avg time: %.3f ms\n", ms);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dBeta); cudaFree(dO);
    free(hQ); free(hK); free(hV); free(hBeta); free(hO); free(hO_ref);
    return 0;
}
