/**
 * RMSNorm (Root Mean Square Normalization)
 * y = x * weight / sqrt(mean(x^2) + eps)
 *
 * Used in LLaMA, Qwen, and other modern LLMs as a simpler alternative to LayerNorm
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

// Warp-level reduction
__device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

// Block-level reduction using shared memory
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

// ========== RMSNorm Forward ==========
// x: [batch, hidden_size], weight: [hidden_size], y: [batch, hidden_size]
__global__ void rmsnorm_forward_kernel(
    const float *__restrict__ x,
    const float *__restrict__ weight,
    float *__restrict__ y,
    int hidden_size, float eps) {

    int batch_idx = blockIdx.x;
    const float *x_row = x + (size_t)batch_idx * hidden_size;
    float *y_row = y + (size_t)batch_idx * hidden_size;

    // Compute sum of squares
    float ss = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = x_row[i];
        ss += val * val;
    }
    ss = block_reduce_sum(ss);

    __shared__ float s_rms_inv;
    if (threadIdx.x == 0) {
        s_rms_inv = rsqrtf(ss / hidden_size + eps);
    }
    __syncthreads();

    // Normalize and scale
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        y_row[i] = x_row[i] * s_rms_inv * weight[i];
    }
}

// ========== RMSNorm Backward ==========
__global__ void rmsnorm_backward_kernel(
    const float *__restrict__ dy,
    const float *__restrict__ x,
    const float *__restrict__ weight,
    float *__restrict__ dx,
    float *__restrict__ dweight,
    int hidden_size, float eps) {

    int batch_idx = blockIdx.x;
    const float *dy_row = dy + (size_t)batch_idx * hidden_size;
    const float *x_row = x + (size_t)batch_idx * hidden_size;
    float *dx_row = dx + (size_t)batch_idx * hidden_size;

    // Recompute RMS
    float ss = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = x_row[i];
        ss += val * val;
    }
    ss = block_reduce_sum(ss);

    __shared__ float s_rms_inv;
    if (threadIdx.x == 0) {
        s_rms_inv = rsqrtf(ss / hidden_size + eps);
    }
    __syncthreads();

    float rms_inv = s_rms_inv;

    // Compute inner product: sum(dy * x * weight)
    float dot_val = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        dot_val += dy_row[i] * x_row[i] * weight[i];
    }
    dot_val = block_reduce_sum(dot_val);

    __shared__ float s_dot;
    if (threadIdx.x == 0) s_dot = dot_val;
    __syncthreads();

    // dx = rms_inv * (dy * weight - x * dot / (ss + eps * hidden_size))
    float scale = s_dot * rms_inv * rms_inv * rms_inv / hidden_size;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        dx_row[i] = rms_inv * (dy_row[i] * weight[i] - x_row[i] * scale);
        // Accumulate dweight (using atomics across batches)
        atomicAdd(&dweight[i], dy_row[i] * x_row[i] * rms_inv);
    }
}

// Host wrappers
void rmsnorm_forward(const float *x, const float *weight, float *y,
                      int batch, int hidden_size, float eps = 1e-6f) {
    int threads = min(1024, hidden_size);
    rmsnorm_forward_kernel<<<batch, threads>>>(x, weight, y, hidden_size, eps);
}

void rmsnorm_backward(const float *dy, const float *x, const float *weight,
                       float *dx, float *dweight,
                       int batch, int hidden_size, float eps = 1e-6f) {
    int threads = min(1024, hidden_size);
    rmsnorm_backward_kernel<<<batch, threads>>>(dy, x, weight, dx, dweight, hidden_size, eps);
}

// ---- Test ----
int main() {
    int batch = 32, hidden = 4096;
    printf("RMSNorm: batch=%d, hidden=%d\n", batch, hidden);

    size_t xSize = (size_t)batch * hidden * sizeof(float);
    size_t wSize = hidden * sizeof(float);

    float *hx = (float *)malloc(xSize);
    float *hw = (float *)malloc(wSize);
    float *hy = (float *)malloc(xSize);
    float *hy_ref = (float *)malloc(xSize);

    srand(42);
    for (int i = 0; i < batch * hidden; i++) hx[i] = ((float)(rand() % 200) - 100) / 100.0f;
    for (int i = 0; i < hidden; i++) hw[i] = 1.0f + ((float)(rand() % 20) - 10) / 100.0f;

    // Reference CPU
    for (int b = 0; b < batch; b++) {
        float ss = 0;
        for (int i = 0; i < hidden; i++) ss += hx[b * hidden + i] * hx[b * hidden + i];
        float rms_inv = 1.0f / sqrtf(ss / hidden + 1e-6f);
        for (int i = 0; i < hidden; i++)
            hy_ref[b * hidden + i] = hx[b * hidden + i] * rms_inv * hw[i];
    }

    float *dx, *dw, *dy;
    CHECK_CUDA(cudaMalloc(&dx, xSize));
    CHECK_CUDA(cudaMalloc(&dw, wSize));
    CHECK_CUDA(cudaMalloc(&dy, xSize));
    CHECK_CUDA(cudaMemcpy(dx, hx, xSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw, hw, wSize, cudaMemcpyHostToDevice));

    rmsnorm_forward(dx, dw, dy, batch, hidden);
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
    for (int i = 0; i < nIter; i++) rmsnorm_forward(dx, dw, dy, batch, hidden);
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
