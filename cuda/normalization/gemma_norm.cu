/**
 * Gemma-style Fused Normalization
 * y = (1 + weight) * x / rms(x)
 *
 * Gemma uses RMSNorm with (1 + w) scaling instead of plain w scaling.
 * This kernel also fuses the norm with a residual add:
 *   x = x + residual
 *   y = (1 + weight) * x / rms(x)
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

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ float block_reduce_sum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

// ========== Gemma RMSNorm: y = (1 + weight) * x * rsqrt(mean(x^2) + eps) ==========
__global__ void gemma_rmsnorm_kernel(
    const float *__restrict__ x,
    const float *__restrict__ weight,
    float *__restrict__ y,
    int hidden_size, float eps) {

    int batch_idx = blockIdx.x;
    const float *x_row = x + (size_t)batch_idx * hidden_size;
    float *y_row = y + (size_t)batch_idx * hidden_size;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = x_row[i];
        ss += val * val;
    }
    ss = block_reduce_sum(ss);

    __shared__ float s_rms_inv;
    if (threadIdx.x == 0) s_rms_inv = rsqrtf(ss / hidden_size + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        y_row[i] = (1.0f + weight[i]) * x_row[i] * s_rms_inv;
    }
}

// ========== Fused Residual + Gemma RMSNorm ==========
// x_out = x + residual
// y = (1 + weight) * x_out * rsqrt(mean(x_out^2) + eps)
__global__ void gemma_fused_residual_rmsnorm_kernel(
    const float *__restrict__ x,
    const float *__restrict__ residual,
    const float *__restrict__ weight,
    float *__restrict__ x_out,     // updated x (for residual stream)
    float *__restrict__ y,
    int hidden_size, float eps) {

    int batch_idx = blockIdx.x;
    const float *x_row = x + (size_t)batch_idx * hidden_size;
    const float *r_row = residual + (size_t)batch_idx * hidden_size;
    float *xo_row = x_out + (size_t)batch_idx * hidden_size;
    float *y_row = y + (size_t)batch_idx * hidden_size;

    // Phase 1: add residual and compute sum of squares
    float ss = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = x_row[i] + r_row[i];
        xo_row[i] = val;  // store for residual stream
        ss += val * val;
    }
    ss = block_reduce_sum(ss);

    __shared__ float s_rms_inv;
    if (threadIdx.x == 0) s_rms_inv = rsqrtf(ss / hidden_size + eps);
    __syncthreads();

    // Phase 2: normalize with (1 + weight) scaling
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        y_row[i] = (1.0f + weight[i]) * xo_row[i] * s_rms_inv;
    }
}

// Host wrappers
void gemma_rmsnorm(const float *x, const float *weight, float *y,
                    int batch, int hidden, float eps = 1e-6f) {
    int threads = min(1024, hidden);
    gemma_rmsnorm_kernel<<<batch, threads>>>(x, weight, y, hidden, eps);
}

void gemma_fused_residual_rmsnorm(const float *x, const float *residual,
                                    const float *weight, float *x_out, float *y,
                                    int batch, int hidden, float eps = 1e-6f) {
    int threads = min(1024, hidden);
    gemma_fused_residual_rmsnorm_kernel<<<batch, threads>>>(
        x, residual, weight, x_out, y, hidden, eps);
}

// ---- Test ----
int main() {
    int batch = 32, hidden = 4096;
    printf("Gemma RMSNorm: batch=%d, hidden=%d\n", batch, hidden);

    size_t xSize = (size_t)batch * hidden * sizeof(float);
    size_t wSize = hidden * sizeof(float);

    float *hx = (float *)malloc(xSize);
    float *hw = (float *)malloc(wSize);
    float *hy = (float *)malloc(xSize);
    float *hy_ref = (float *)malloc(xSize);

    srand(42);
    for (int i = 0; i < batch * hidden; i++) hx[i] = ((float)(rand() % 200) - 100) / 100.0f;
    for (int i = 0; i < hidden; i++) hw[i] = ((float)(rand() % 20) - 10) / 100.0f;  // small values near 0

    // CPU reference: y = (1 + w) * x / rms(x)
    for (int b = 0; b < batch; b++) {
        float ss = 0;
        for (int i = 0; i < hidden; i++) ss += hx[b * hidden + i] * hx[b * hidden + i];
        float rms_inv = 1.0f / sqrtf(ss / hidden + 1e-6f);
        for (int i = 0; i < hidden; i++)
            hy_ref[b * hidden + i] = (1.0f + hw[i]) * hx[b * hidden + i] * rms_inv;
    }

    float *dx, *dw, *dy;
    CHECK_CUDA(cudaMalloc(&dx, xSize));
    CHECK_CUDA(cudaMalloc(&dw, wSize));
    CHECK_CUDA(cudaMalloc(&dy, xSize));
    CHECK_CUDA(cudaMemcpy(dx, hx, xSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw, hw, wSize, cudaMemcpyHostToDevice));

    gemma_rmsnorm(dx, dw, dy, batch, hidden);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hy, dy, xSize, cudaMemcpyDeviceToHost));

    float maxErr = 0;
    for (int i = 0; i < batch * hidden; i++) {
        float err = fabsf(hy[i] - hy_ref[i]);
        if (err > maxErr) maxErr = err;
    }
    printf("Max error: %e -> %s\n", maxErr, maxErr < 1e-4f ? "PASS" : "FAIL");

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    int nIter = 100;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < nIter; i++) gemma_rmsnorm(dx, dw, dy, batch, hidden);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= nIter;
    double gbps = 2.0 * batch * hidden * sizeof(float) / (ms * 1e-3) / 1e9;
    printf("Avg time: %.4f ms | %.2f GB/s\n", ms, gbps);

    cudaFree(dx); cudaFree(dw); cudaFree(dy);
    free(hx); free(hw); free(hy); free(hy_ref);
    return 0;
}
