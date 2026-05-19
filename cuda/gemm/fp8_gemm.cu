/**
 * FP8 GEMM with Per-Tensor and Groupwise Scaling
 * C = scale_a * scale_b * (A_fp8 @ B_fp8)
 * Requires SM89+ (Hopper/Ada Lovelace)
 *
 * Two modes:
 *   1) Per-tensor scaling: single scale factor per tensor
 *   2) Groupwise scaling:  one scale per group of G elements along K
 */
#include <cuda.h>
#include <cuda_fp8.h>
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

// ========== Per-Tensor Scaled FP8 GEMM ==========
// Each thread computes a small tile of C using dot products in FP8 accumulated in FP32
__global__ void fp8_gemm_per_tensor_kernel(
    const __nv_fp8_e4m3 *__restrict__ A,
    const __nv_fp8_e4m3 *__restrict__ B,
    float *__restrict__ C,
    int M, int N, int K,
    float scale_a, float scale_b) {

    const int TILE = 16;
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K)
            ? float(A[row * K + aCol]) * scale_a
            : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < K && col < N)
            ? float(B[bRow * N + col]) * scale_b
            : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ========== Groupwise Scaled FP8 GEMM ==========
// scale_a: [M, K/G], scale_b: [K/G, N]
__global__ void fp8_gemm_groupwise_kernel(
    const __nv_fp8_e4m3 *__restrict__ A,
    const __nv_fp8_e4m3 *__restrict__ B,
    float *__restrict__ C,
    int M, int N, int K, int G,
    const float *__restrict__ scale_a,   // [M, num_groups]
    const float *__restrict__ scale_b) { // [num_groups, N]

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    int num_groups = (K + G - 1) / G;
    float acc = 0.0f;

    for (int g = 0; g < num_groups; g++) {
        int k_start = g * G;
        int k_end = min(k_start + G, K);
        float sa = scale_a[row * num_groups + g];
        float sb = scale_b[g * N + col];
        float group_acc = 0.0f;

        for (int k = k_start; k < k_end; k++) {
            float a_val = float(A[row * K + k]);
            float b_val = float(B[k * N + col]);
            group_acc += a_val * b_val;
        }
        acc += group_acc * sa * sb;
    }
    C[row * N + col] = acc;
}

// ========== Host Wrappers ==========
void fp8_gemm_per_tensor(const __nv_fp8_e4m3 *A, const __nv_fp8_e4m3 *B,
                          float *C, int M, int N, int K,
                          float scale_a, float scale_b) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    fp8_gemm_per_tensor_kernel<<<grid, block>>>(A, B, C, M, N, K, scale_a, scale_b);
}

void fp8_gemm_groupwise(const __nv_fp8_e4m3 *A, const __nv_fp8_e4m3 *B,
                          float *C, int M, int N, int K, int G,
                          const float *scale_a, const float *scale_b) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    fp8_gemm_groupwise_kernel<<<grid, block>>>(A, B, C, M, N, K, G, scale_a, scale_b);
}

// ========== Quantize FP32 -> FP8 with scale ==========
__global__ void quantize_to_fp8_kernel(const float *input, __nv_fp8_e4m3 *output,
                                        float *scale_out, int n) {
    // Compute absmax
    __shared__ float smax;
    if (threadIdx.x == 0) smax = 0;
    __syncthreads();

    float local_max = 0;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        local_max = fmaxf(local_max, fabsf(input[i]));
    }
    atomicMax((int *)&smax, __float_as_int(local_max));
    __syncthreads();

    float scale = smax / 448.0f;  // FP8 E4M3 max value ~448
    if (scale < 1e-12f) scale = 1e-12f;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *scale_out = scale;
    }

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        float val = input[i] / scale;
        output[i] = __nv_fp8_e4m3(val);
    }
}

// ---- Test & Benchmark ----
int main(int argc, char **argv) {
    int M = 2048, N = 2048, K = 2048;
    if (argc >= 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }

    printf("FP8 GEMM (Per-tensor & Groupwise): M=%d, N=%d, K=%d\n", M, N, K);

    // Allocate and init FP32 host data
    size_t sizeA = (size_t)M * K;
    size_t sizeB = (size_t)K * N;
    float *hA = (float *)malloc(sizeA * sizeof(float));
    float *hB = (float *)malloc(sizeB * sizeof(float));
    float *hC = (float *)malloc((size_t)M * N * sizeof(float));

    srand(42);
    for (size_t i = 0; i < sizeA; i++) hA[i] = ((float)(rand() % 200) - 100) / 100.0f;
    for (size_t i = 0; i < sizeB; i++) hB[i] = ((float)(rand() % 200) - 100) / 100.0f;

    // Device memory
    float *dA_f32, *dB_f32;
    __nv_fp8_e4m3 *dA_fp8, *dB_fp8;
    float *dC, *dScaleA, *dScaleB;

    CHECK_CUDA(cudaMalloc(&dA_f32, sizeA * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB_f32, sizeB * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dA_fp8, sizeA * sizeof(__nv_fp8_e4m3)));
    CHECK_CUDA(cudaMalloc(&dB_fp8, sizeB * sizeof(__nv_fp8_e4m3)));
    CHECK_CUDA(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dScaleA, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dScaleB, sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA_f32, hA, sizeA * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_f32, hB, sizeB * sizeof(float), cudaMemcpyHostToDevice));

    // Quantize to FP8
    quantize_to_fp8_kernel<<<1, 256>>>(dA_f32, dA_fp8, dScaleA, sizeA);
    quantize_to_fp8_kernel<<<1, 256>>>(dB_f32, dB_fp8, dScaleB, sizeB);
    CHECK_CUDA(cudaDeviceSynchronize());

    float hScaleA, hScaleB;
    CHECK_CUDA(cudaMemcpy(&hScaleA, dScaleA, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&hScaleB, dScaleB, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Scale A: %e, Scale B: %e\n", hScaleA, hScaleB);

    // ---- Per-tensor test ----
    CHECK_CUDA(cudaMemset(dC, 0, (size_t)M * N * sizeof(float)));
    fp8_gemm_per_tensor(dA_fp8, dB_fp8, dC, M, N, K, hScaleA, hScaleB);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark per-tensor
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int nIter = 20;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < nIter; i++) {
        fp8_gemm_per_tensor(dA_fp8, dB_fp8, dC, M, N, K, hScaleA, hScaleB);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= nIter;
    double tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12;
    printf("[Per-tensor] Avg time: %.3f ms | %.2f TFLOPS\n", ms, tflops);

    // Correctness (small block)
    CHECK_CUDA(cudaMemcpy(hC, dC, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
    int check = min(M, 32);
    float maxErr = 0;
    for (int i = 0; i < check; i++) {
        for (int j = 0; j < check; j++) {
            float ref = 0;
            for (int k = 0; k < K; k++) ref += hA[i * K + k] * hB[k * N + j];
            float err = fabsf(hC[i * N + j] - ref) / (fabsf(ref) + 1e-6f);
            if (err > maxErr) maxErr = err;
        }
    }
    printf("[Per-tensor] Max relative error: %e -> %s\n", maxErr, maxErr < 0.1f ? "PASS" : "FAIL");

    // Cleanup
    cudaFree(dA_f32); cudaFree(dB_f32); cudaFree(dA_fp8); cudaFree(dB_fp8);
    cudaFree(dC); cudaFree(dScaleA); cudaFree(dScaleB);
    free(hA); free(hB); free(hC);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
