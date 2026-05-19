/**
 * Gated Softmax Attention (Qwen3NeXt)
 *
 * Attention with gating on the softmax output:
 *   attn_weights = softmax(Q @ K^T / sqrt(d)) * gate
 *   output = attn_weights @ V
 *
 * The gate is a learnable per-head scalar or vector that modulates attention.
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

// ========== Gated Softmax Attention Kernel ==========
// Q, K, V: [batch, num_heads, seq_len, head_dim]
// gate: [num_heads] (per-head gating scalar, applied as sigmoid(gate))
// output: [batch, num_heads, seq_len, head_dim]
__global__ void gated_softmax_attention_kernel(
    const float *__restrict__ Q,
    const float *__restrict__ K,
    const float *__restrict__ V,
    const float *__restrict__ gate,   // [num_heads]
    float *__restrict__ output,
    int seq_len, int head_dim, float scale) {

    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int query_idx = blockIdx.x;

    // Gate value: sigmoid for soft gating
    float g = 1.0f / (1.0f + expf(-gate[head_idx]));

    extern __shared__ float smem[];
    float *attn_scores = smem;  // [seq_len]

    size_t qkv_stride = (size_t)seq_len * head_dim;
    size_t batch_head_off = (size_t)batch_idx * gridDim.y * qkv_stride +
                            (size_t)head_idx * qkv_stride;

    const float *q_row = Q + batch_head_off + (size_t)query_idx * head_dim;

    // Step 1: Compute Q @ K^T for this query position
    float max_score = -FLT_MAX;
    for (int kv_idx = threadIdx.x; kv_idx < seq_len; kv_idx += blockDim.x) {
        const float *k_row = K + batch_head_off + (size_t)kv_idx * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_row[d] * k_row[d];
        }
        dot *= scale;

        // Causal mask
        if (kv_idx > query_idx) dot = -FLT_MAX;

        attn_scores[kv_idx] = dot;
        max_score = fmaxf(max_score, dot);
    }
    __syncthreads();

    // Block-wide max reduction
    __shared__ float s_max;
    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x)
        local_max = fmaxf(local_max, attn_scores[i]);

    // Simple warp reduction
    local_max = warp_reduce_max(local_max);
    if (threadIdx.x % 32 == 0) smem[seq_len + threadIdx.x / 32] = local_max;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? smem[seq_len + threadIdx.x] : -FLT_MAX;
        v = warp_reduce_max(v);
        if (threadIdx.x == 0) s_max = v;
    }
    __syncthreads();

    // Step 2: Softmax with gating
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float val = expf(attn_scores[i] - s_max);
        attn_scores[i] = val;
        sum_exp += val;
    }
    __syncthreads();

    // Sum reduction
    __shared__ float s_sum;
    float local_sum = 0;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x)
        local_sum += attn_scores[i];
    local_sum = warp_reduce_sum(local_sum);
    if (threadIdx.x % 32 == 0) smem[seq_len + threadIdx.x / 32] = local_sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? smem[seq_len + threadIdx.x] : 0;
        v = warp_reduce_sum(v);
        if (threadIdx.x == 0) s_sum = v;
    }
    __syncthreads();

    float inv_sum = 1.0f / (s_sum + 1e-6f);
    // Apply gate after softmax
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        attn_scores[i] = attn_scores[i] * inv_sum * g;
    }
    __syncthreads();

    // Step 3: attn_weights @ V
    float *out_row = output + batch_head_off + (size_t)query_idx * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float val = 0.0f;
        for (int kv_idx = 0; kv_idx < seq_len; kv_idx++) {
            val += attn_scores[kv_idx] * V[batch_head_off + (size_t)kv_idx * head_dim + d];
        }
        out_row[d] = val;
    }
}

// Host wrapper
void gated_softmax_attention(const float *Q, const float *K, const float *V,
                              const float *gate, float *output,
                              int batch, int num_heads, int seq_len, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    dim3 grid(seq_len, num_heads, batch);
    int threads = min(256, seq_len);
    int smem = (seq_len + 32) * sizeof(float);
    gated_softmax_attention_kernel<<<grid, threads, smem>>>(
        Q, K, V, gate, output, seq_len, head_dim, scale);
}

// ---- Test ----
int main() {
    int batch = 2, heads = 8, seq = 64, dim = 64;
    printf("Gated Softmax Attention: batch=%d, heads=%d, seq=%d, dim=%d\n",
           batch, heads, seq, dim);

    size_t qkvSize = (size_t)batch * heads * seq * dim * sizeof(float);
    size_t gateSize = heads * sizeof(float);

    float *hQ = (float *)malloc(qkvSize);
    float *hK = (float *)malloc(qkvSize);
    float *hV = (float *)malloc(qkvSize);
    float *hGate = (float *)malloc(gateSize);
    float *hO = (float *)malloc(qkvSize);

    srand(42);
    for (size_t i = 0; i < (size_t)batch * heads * seq * dim; i++) {
        hQ[i] = ((float)(rand() % 200) - 100) / 1000.0f;
        hK[i] = ((float)(rand() % 200) - 100) / 1000.0f;
        hV[i] = ((float)(rand() % 200) - 100) / 1000.0f;
    }
    for (int i = 0; i < heads; i++) hGate[i] = ((float)(rand() % 200) - 100) / 100.0f;

    float *dQ, *dK, *dV, *dGate, *dO;
    CHECK_CUDA(cudaMalloc(&dQ, qkvSize));
    CHECK_CUDA(cudaMalloc(&dK, qkvSize));
    CHECK_CUDA(cudaMalloc(&dV, qkvSize));
    CHECK_CUDA(cudaMalloc(&dGate, gateSize));
    CHECK_CUDA(cudaMalloc(&dO, qkvSize));
    CHECK_CUDA(cudaMemcpy(dQ, hQ, qkvSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, hK, qkvSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, hV, qkvSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dGate, hGate, gateSize, cudaMemcpyHostToDevice));

    gated_softmax_attention(dQ, dK, dV, dGate, dO, batch, heads, seq, dim);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hO, dO, qkvSize, cudaMemcpyDeviceToHost));

    // Basic sanity: output should not be NaN or Inf
    bool valid = true;
    for (size_t i = 0; i < (size_t)batch * heads * seq * dim; i++) {
        if (isnan(hO[i]) || isinf(hO[i])) { valid = false; break; }
    }
    printf("Output valid (no NaN/Inf): %s\n", valid ? "PASS" : "FAIL");

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    int nIter = 50;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < nIter; i++)
        gated_softmax_attention(dQ, dK, dV, dGate, dO, batch, heads, seq, dim);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= nIter;
    printf("Avg time: %.3f ms\n", ms);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dGate); cudaFree(dO);
    free(hQ); free(hK); free(hV); free(hGate); free(hO);
    return 0;
}
