/**
 * Fused Gated Activations: SiLU (SwiGLU) and GELU (GeGLU)
 *
 * SwiGLU: output = SiLU(gate) * up = (gate * sigmoid(gate)) * up
 * GeGLU:  output = GELU(gate) * up
 *
 * Input: [batch, 2 * hidden] split into gate and up projections
 * Output: [batch, hidden]
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

// SiLU: x * sigmoid(x)
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// GELU (tanh approximation)
__device__ __forceinline__ float gelu(float x) {
    const float c = 0.7978845608f; // sqrt(2/pi)
    const float k = 0.044715f;
    float inner = c * (x + k * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// ========== SwiGLU: output = SiLU(gate) * up ==========
// input: [batch, 2*hidden], output: [batch, hidden]
__global__ void swiglu_kernel(
    const float *__restrict__ input,
    float *__restrict__ output,
    int hidden_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / hidden_size;
    int h_idx = idx % hidden_size;
    int total = gridDim.x * blockDim.x;

    for (int i = idx; i < blockIdx.y * hidden_size + hidden_size; i += total) {
        // Recalculate indices for grid-stride
    }

    if (idx >= blockIdx.y) return; // placeholder guard

    // Use simple flat indexing
    int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // We'll use a simpler kernel:
}

// Simpler flat kernel for SwiGLU
__global__ void swiglu_flat_kernel(
    const float *__restrict__ gate,   // [batch, hidden]
    const float *__restrict__ up,     // [batch, hidden]
    float *__restrict__ output,       // [batch, hidden]
    int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = silu(gate[idx]) * up[idx];
}

// Fused version: input is interleaved [gate_0, up_0, gate_1, up_1, ...]
__global__ void swiglu_fused_kernel(
    const float *__restrict__ input,   // [batch, 2 * hidden]
    float *__restrict__ output,        // [batch, hidden]
    int batch_size, int hidden_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * hidden_size) return;

    int b = idx / hidden_size;
    int h = idx % hidden_size;

    float gate_val = input[b * 2 * hidden_size + h];
    float up_val = input[b * 2 * hidden_size + hidden_size + h];
    output[idx] = silu(gate_val) * up_val;
}

// ========== GeGLU: output = GELU(gate) * up ==========
__global__ void geglu_fused_kernel(
    const float *__restrict__ input,   // [batch, 2 * hidden]
    float *__restrict__ output,        // [batch, hidden]
    int batch_size, int hidden_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * hidden_size) return;

    int b = idx / hidden_size;
    int h = idx % hidden_size;

    float gate_val = input[b * 2 * hidden_size + h];
    float up_val = input[b * 2 * hidden_size + hidden_size + h];
    output[idx] = gelu(gate_val) * up_val;
}

// ========== Backward kernels ==========
// SwiGLU backward
__global__ void swiglu_backward_kernel(
    const float *__restrict__ grad_output,  // [batch, hidden]
    const float *__restrict__ input,        // [batch, 2*hidden]
    float *__restrict__ grad_input,         // [batch, 2*hidden]
    int batch_size, int hidden_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * hidden_size) return;

    int b = idx / hidden_size;
    int h = idx % hidden_size;

    float gate_val = input[b * 2 * hidden_size + h];
    float up_val = input[b * 2 * hidden_size + hidden_size + h];
    float go = grad_output[idx];

    float sig = 1.0f / (1.0f + expf(-gate_val));
    float silu_val = gate_val * sig;

    // d_up = SiLU(gate) * grad_output
    grad_input[b * 2 * hidden_size + hidden_size + h] = silu_val * go;

    // d_gate = up * grad_output * d_SiLU(gate)
    // d_SiLU(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    //           = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    float dsilu = sig * (1.0f + gate_val * (1.0f - sig));
    grad_input[b * 2 * hidden_size + h] = up_val * go * dsilu;
}

// Host wrappers
void swiglu_forward(const float *input, float *output,
                     int batch, int hidden) {
    int n = batch * hidden;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    swiglu_fused_kernel<<<blocks, threads>>>(input, output, batch, hidden);
}

void geglu_forward(const float *input, float *output,
                    int batch, int hidden) {
    int n = batch * hidden;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    geglu_fused_kernel<<<blocks, threads>>>(input, output, batch, hidden);
}

// ---- Test ----
int main() {
    int batch = 32, hidden = 4096;
    printf("Fused Gated Activations: batch=%d, hidden=%d\n", batch, hidden);

    size_t inSize = (size_t)batch * 2 * hidden * sizeof(float);
    size_t outSize = (size_t)batch * hidden * sizeof(float);

    float *h_input = (float *)malloc(inSize);
    float *h_out_swiglu = (float *)malloc(outSize);
    float *h_out_geglu = (float *)malloc(outSize);
    float *h_ref_swiglu = (float *)malloc(outSize);
    float *h_ref_geglu = (float *)malloc(outSize);

    srand(42);
    for (size_t i = 0; i < (size_t)batch * 2 * hidden; i++)
        h_input[i] = ((float)(rand() % 200) - 100) / 100.0f;

    // CPU reference
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < hidden; h++) {
            float gate = h_input[b * 2 * hidden + h];
            float up = h_input[b * 2 * hidden + hidden + h];
            // SiLU
            float sig = 1.0f / (1.0f + expf(-gate));
            h_ref_swiglu[b * hidden + h] = (gate * sig) * up;
            // GELU
            float c = 0.7978845608f, k = 0.044715f;
            float inner = c * (gate + k * gate * gate * gate);
            h_ref_geglu[b * hidden + h] = (0.5f * gate * (1.0f + tanhf(inner))) * up;
        }
    }

    float *d_input, *d_out;
    CHECK_CUDA(cudaMalloc(&d_input, inSize));
    CHECK_CUDA(cudaMalloc(&d_out, outSize));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, inSize, cudaMemcpyHostToDevice));

    // Test SwiGLU
    swiglu_forward(d_input, d_out, batch, hidden);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out_swiglu, d_out, outSize, cudaMemcpyDeviceToHost));

    float maxErr = 0;
    for (size_t i = 0; i < (size_t)batch * hidden; i++) {
        float err = fabsf(h_out_swiglu[i] - h_ref_swiglu[i]);
        if (err > maxErr) maxErr = err;
    }
    printf("[SwiGLU] Max error: %e -> %s\n", maxErr, maxErr < 1e-5f ? "PASS" : "FAIL");

    // Test GeGLU
    geglu_forward(d_input, d_out, batch, hidden);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out_geglu, d_out, outSize, cudaMemcpyDeviceToHost));

    maxErr = 0;
    for (size_t i = 0; i < (size_t)batch * hidden; i++) {
        float err = fabsf(h_out_geglu[i] - h_ref_geglu[i]);
        if (err > maxErr) maxErr = err;
    }
    printf("[GeGLU]  Max error: %e -> %s\n", maxErr, maxErr < 1e-5f ? "PASS" : "FAIL");

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    int nIter = 100;

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < nIter; i++) swiglu_forward(d_input, d_out, batch, hidden);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= nIter;
    double gbps = ((size_t)batch * 3 * hidden * sizeof(float)) / (ms * 1e-3) / 1e9;
    printf("[SwiGLU] Avg: %.4f ms | %.2f GB/s\n", ms, gbps);

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < nIter; i++) geglu_forward(d_input, d_out, batch, hidden);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= nIter;
    printf("[GeGLU]  Avg: %.4f ms | %.2f GB/s\n", ms, gbps);

    cudaFree(d_input); cudaFree(d_out);
    free(h_input); free(h_out_swiglu); free(h_out_geglu);
    free(h_ref_swiglu); free(h_ref_geglu);
    return 0;
}
