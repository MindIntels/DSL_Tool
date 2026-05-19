/**
 * RoPE - Rotary Position Embeddings (LLaMA style)
 *
 * Implements standard RoPE and LLaMA 3.1 extended RoPE with:
 *   - Frequency scaling for context extension
 *   - NTK-aware interpolation
 *
 * Applies rotation: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
 * for each consecutive pair in the head dimension
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

// ========== Standard RoPE ==========
// x: [batch, seq_len, num_heads, head_dim]
// Applied in-place
__global__ void rope_forward_kernel(
    float *__restrict__ x,
    const int *__restrict__ positions,   // [batch, seq_len] or NULL for sequential
    int seq_len, int num_heads, int head_dim,
    float base, float scale) {

    int batch_idx = blockIdx.z;
    int seq_idx = blockIdx.y;
    int head_idx = blockIdx.x;
    int half_dim = head_dim / 2;

    int pos = positions ? positions[batch_idx * seq_len + seq_idx] : seq_idx;

    for (int d = threadIdx.x; d < half_dim; d += blockDim.x) {
        float freq = 1.0f / powf(base, (float)(2 * d) / head_dim);
        float theta = (float)pos * freq * scale;
        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);

        size_t offset = ((size_t)batch_idx * seq_len * num_heads * head_dim +
                         (size_t)seq_idx * num_heads * head_dim +
                         (size_t)head_idx * head_dim);

        float x0 = x[offset + d];
        float x1 = x[offset + d + half_dim];

        x[offset + d]            = x0 * cos_theta - x1 * sin_theta;
        x[offset + d + half_dim] = x0 * sin_theta + x1 * cos_theta;
    }
}

// ========== LLaMA 3.1 RoPE with frequency scaling ==========
// Uses different scaling factors for different frequency bands:
//   - Low frequencies: keep original (no scaling)
//   - High frequencies: scale by factor
//   - Transition band: smooth interpolation
__global__ void rope_llama31_kernel(
    float *__restrict__ x,
    const int *__restrict__ positions,
    int seq_len, int num_heads, int head_dim,
    float base,
    float scaling_factor,
    float low_freq_factor,       // default 1.0
    float high_freq_factor,      // default 4.0
    int original_max_position) { // original context length (e.g. 8192)

    int batch_idx = blockIdx.z;
    int seq_idx = blockIdx.y;
    int head_idx = blockIdx.x;
    int half_dim = head_dim / 2;

    int pos = positions ? positions[batch_idx * seq_len + seq_idx] : seq_idx;
    float old_context_len = (float)original_max_position;

    for (int d = threadIdx.x; d < half_dim; d += blockDim.x) {
        float base_freq = 1.0f / powf(base, (float)(2 * d) / head_dim);
        float wavelength = 2.0f * M_PI / base_freq;

        float freq;
        if (wavelength < low_freq_factor * old_context_len) {
            // High frequency: apply full scaling
            freq = base_freq / scaling_factor;
        } else if (wavelength > high_freq_factor * old_context_len) {
            // Low frequency: keep original
            freq = base_freq;
        } else {
            // Transition band: smooth interpolation
            float smooth = (old_context_len / wavelength - low_freq_factor) /
                          (high_freq_factor - low_freq_factor);
            freq = (1.0f - smooth) * base_freq / scaling_factor + smooth * base_freq;
        }

        float theta = (float)pos * freq;
        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);

        size_t offset = ((size_t)batch_idx * seq_len * num_heads * head_dim +
                         (size_t)seq_idx * num_heads * head_dim +
                         (size_t)head_idx * head_dim);

        float x0 = x[offset + d];
        float x1 = x[offset + d + half_dim];

        x[offset + d]            = x0 * cos_theta - x1 * sin_theta;
        x[offset + d + half_dim] = x0 * sin_theta + x1 * cos_theta;
    }
}

// ========== Precompute cos/sin cache ==========
__global__ void rope_precompute_freqs_kernel(
    float *__restrict__ cos_cache,   // [max_seq, head_dim/2]
    float *__restrict__ sin_cache,   // [max_seq, head_dim/2]
    int max_seq_len, int head_dim, float base) {

    int pos = blockIdx.x;
    int half_dim = head_dim / 2;

    for (int d = threadIdx.x; d < half_dim; d += blockDim.x) {
        float freq = 1.0f / powf(base, (float)(2 * d) / head_dim);
        float theta = (float)pos * freq;
        cos_cache[pos * half_dim + d] = cosf(theta);
        sin_cache[pos * half_dim + d] = sinf(theta);
    }
}

// ========== RoPE with precomputed cache (faster for inference) ==========
__global__ void rope_cached_kernel(
    float *__restrict__ x,
    const float *__restrict__ cos_cache,
    const float *__restrict__ sin_cache,
    const int *__restrict__ positions,
    int seq_len, int num_heads, int head_dim) {

    int batch_idx = blockIdx.z;
    int seq_idx = blockIdx.y;
    int head_idx = blockIdx.x;
    int half_dim = head_dim / 2;

    int pos = positions ? positions[batch_idx * seq_len + seq_idx] : seq_idx;

    for (int d = threadIdx.x; d < half_dim; d += blockDim.x) {
        float cos_val = cos_cache[pos * half_dim + d];
        float sin_val = sin_cache[pos * half_dim + d];

        size_t offset = ((size_t)batch_idx * seq_len * num_heads * head_dim +
                         (size_t)seq_idx * num_heads * head_dim +
                         (size_t)head_idx * head_dim);

        float x0 = x[offset + d];
        float x1 = x[offset + d + half_dim];

        x[offset + d]            = x0 * cos_val - x1 * sin_val;
        x[offset + d + half_dim] = x0 * sin_val + x1 * cos_val;
    }
}

// Host wrappers
void rope_forward(float *x, const int *positions,
                   int batch, int seq_len, int num_heads, int head_dim,
                   float base = 10000.0f, float scale = 1.0f) {
    dim3 grid(num_heads, seq_len, batch);
    int threads = min(256, head_dim / 2);
    rope_forward_kernel<<<grid, threads>>>(x, positions, seq_len, num_heads, head_dim, base, scale);
}

void rope_llama31(float *x, const int *positions,
                   int batch, int seq_len, int num_heads, int head_dim,
                   float base = 500000.0f, float scaling_factor = 8.0f,
                   float low_freq = 1.0f, float high_freq = 4.0f,
                   int orig_max_pos = 8192) {
    dim3 grid(num_heads, seq_len, batch);
    int threads = min(256, head_dim / 2);
    rope_llama31_kernel<<<grid, threads>>>(x, positions, seq_len, num_heads, head_dim,
                                            base, scaling_factor, low_freq, high_freq, orig_max_pos);
}

// ---- Test ----
int main() {
    int batch = 2, seq_len = 128, num_heads = 32, head_dim = 128;
    printf("RoPE: batch=%d, seq=%d, heads=%d, dim=%d\n", batch, seq_len, num_heads, head_dim);

    size_t total = (size_t)batch * seq_len * num_heads * head_dim;
    size_t totalBytes = total * sizeof(float);

    float *hx = (float *)malloc(totalBytes);
    float *hx_copy = (float *)malloc(totalBytes);
    float *hx_ref = (float *)malloc(totalBytes);

    srand(42);
    for (size_t i = 0; i < total; i++) {
        hx[i] = ((float)(rand() % 200) - 100) / 100.0f;
        hx_copy[i] = hx[i];
        hx_ref[i] = hx[i];
    }

    // CPU reference for standard RoPE
    int half_dim = head_dim / 2;
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int h = 0; h < num_heads; h++) {
                size_t off = ((size_t)b * seq_len * num_heads * head_dim +
                              (size_t)s * num_heads * head_dim +
                              (size_t)h * head_dim);
                for (int d = 0; d < half_dim; d++) {
                    float freq = 1.0f / powf(10000.0f, (float)(2 * d) / head_dim);
                    float theta = (float)s * freq;
                    float cos_t = cosf(theta), sin_t = sinf(theta);
                    float x0 = hx_ref[off + d];
                    float x1 = hx_ref[off + d + half_dim];
                    hx_ref[off + d]            = x0 * cos_t - x1 * sin_t;
                    hx_ref[off + d + half_dim] = x0 * sin_t + x1 * cos_t;
                }
            }
        }
    }

    // GPU standard RoPE
    float *dx;
    CHECK_CUDA(cudaMalloc(&dx, totalBytes));
    CHECK_CUDA(cudaMemcpy(dx, hx, totalBytes, cudaMemcpyHostToDevice));

    rope_forward(dx, nullptr, batch, seq_len, num_heads, head_dim);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hx, dx, totalBytes, cudaMemcpyDeviceToHost));

    float maxErr = 0;
    for (size_t i = 0; i < total; i++) {
        float err = fabsf(hx[i] - hx_ref[i]);
        if (err > maxErr) maxErr = err;
    }
    printf("[Standard RoPE] Max error: %e -> %s\n", maxErr, maxErr < 1e-4f ? "PASS" : "FAIL");

    // GPU LLaMA 3.1 RoPE
    CHECK_CUDA(cudaMemcpy(dx, hx_copy, totalBytes, cudaMemcpyHostToDevice));
    rope_llama31(dx, nullptr, batch, seq_len, num_heads, head_dim);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hx, dx, totalBytes, cudaMemcpyDeviceToHost));
    printf("[LLaMA 3.1 RoPE] C[0]=%f (sanity check, no CPU ref for extended)\n", hx[0]);

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    int nIter = 100;

    CHECK_CUDA(cudaMemcpy(dx, hx_copy, totalBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < nIter; i++)
        rope_forward(dx, nullptr, batch, seq_len, num_heads, head_dim);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= nIter;
    double gbps = 2.0 * total * sizeof(float) / (ms * 1e-3) / 1e9;
    printf("[Standard RoPE] Avg: %.4f ms | %.2f GB/s\n", ms, gbps);

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < nIter; i++)
        rope_llama31(dx, nullptr, batch, seq_len, num_heads, head_dim);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= nIter;
    printf("[LLaMA 3.1 RoPE] Avg: %.4f ms | %.2f GB/s\n", ms, gbps);

    cudaFree(dx);
    free(hx); free(hx_copy); free(hx_ref);
    return 0;
}
