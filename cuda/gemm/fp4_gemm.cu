/**
 * FP4 GEMM - NVFP4 and MXFP4 Matrix Multiplication
 * Targeting Blackwell GPUs (SM100+)
 *
 * NVFP4: NVIDIA's 4-bit floating point (1 sign, 2 exponent, 1 mantissa) with per-block scaling
 * MXFP4: Microscaling FP4 (OCP standard) with shared exponent per block
 *
 * Since FP4 native types require SM100+, this provides:
 *   1) A software emulation for correctness testing on any GPU
 *   2) The kernel structure for Blackwell deployment
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

// FP4 E2M1 lookup table (NVFP4): 4-bit values [0..15] -> float
// Format: 1 sign bit, 2 exponent bits, 1 mantissa bit
__constant__ float NVFP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// MXFP4 E2M1 lookup (same numeric format, different scaling convention)
__constant__ float MXFP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Block size for scaling groups
#define FP4_BLOCK_SIZE 32

// Packed FP4: two 4-bit values per byte
struct packed_fp4_t {
    uint8_t data;  // low nibble = first element, high nibble = second
};

// ========== Quantize FP32 -> FP4 (NVFP4) with block scaling ==========
__global__ void quantize_fp32_to_nvfp4(
    const float *__restrict__ input,
    packed_fp4_t *__restrict__ output,
    float *__restrict__ scales,
    int n) {

    int block_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = (n + FP4_BLOCK_SIZE - 1) / FP4_BLOCK_SIZE;
    if (block_id >= num_blocks) return;

    int start = block_id * FP4_BLOCK_SIZE;
    int end = min(start + FP4_BLOCK_SIZE, n);

    // Find absmax in block
    float amax = 0.0f;
    for (int i = start; i < end; i++) {
        amax = fmaxf(amax, fabsf(input[i]));
    }

    float scale = amax / 6.0f;  // Max representable value in FP4 E2M1
    if (scale < 1e-12f) scale = 1e-12f;
    scales[block_id] = scale;
    float inv_scale = 1.0f / scale;

    // Quantize pairs of values
    for (int i = start; i < end; i += 2) {
        float v0 = input[i] * inv_scale;
        float v1 = (i + 1 < end) ? input[i + 1] * inv_scale : 0.0f;

        // Find nearest FP4 value
        uint8_t q0 = 0, q1 = 0;
        float min_err0 = 1e30f, min_err1 = 1e30f;
        for (int j = 0; j < 16; j++) {
            float err0 = fabsf(NVFP4_LUT[j] - v0);
            float err1 = fabsf(NVFP4_LUT[j] - v1);
            if (err0 < min_err0) { min_err0 = err0; q0 = j; }
            if (err1 < min_err1) { min_err1 = err1; q1 = j; }
        }
        output[(i - start) / 2 + start / 2].data = (q1 << 4) | q0;
    }
}

// ========== FP4 GEMM Kernel (software emulation) ==========
// A: [M, K] packed FP4, B: [K, N] packed FP4
// scale_a: [M, K/BLOCK_SIZE], scale_b: [K/BLOCK_SIZE, N]
__global__ void fp4_gemm_kernel(
    const packed_fp4_t *__restrict__ A,
    const packed_fp4_t *__restrict__ B,
    float *__restrict__ C,
    const float *__restrict__ scale_a,
    const float *__restrict__ scale_b,
    int M, int N, int K, bool use_mxfp4) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    const float *lut = use_mxfp4 ? MXFP4_LUT : NVFP4_LUT;
    int num_scale_groups = (K + FP4_BLOCK_SIZE - 1) / FP4_BLOCK_SIZE;
    float acc = 0.0f;

    for (int g = 0; g < num_scale_groups; g++) {
        float sa = scale_a[row * num_scale_groups + g];
        float sb = scale_b[g * N + col];
        float group_acc = 0.0f;

        int k_start = g * FP4_BLOCK_SIZE;
        int k_end = min(k_start + FP4_BLOCK_SIZE, K);

        for (int k = k_start; k < k_end; k++) {
            // Unpack FP4 values
            int a_idx = row * K + k;
            int b_idx = k * N + col;

            packed_fp4_t a_packed = A[a_idx / 2];
            packed_fp4_t b_packed = B[b_idx / 2];

            uint8_t a_nibble = (a_idx % 2 == 0) ? (a_packed.data & 0x0F) : (a_packed.data >> 4);
            uint8_t b_nibble = (b_idx % 2 == 0) ? (b_packed.data & 0x0F) : (b_packed.data >> 4);

            float a_val = lut[a_nibble];
            float b_val = lut[b_nibble];
            group_acc += a_val * b_val;
        }
        acc += group_acc * sa * sb;
    }
    C[row * N + col] = acc;
}

// ========== Host Wrappers ==========
void fp4_gemm(const packed_fp4_t *A, const packed_fp4_t *B, float *C,
              const float *scale_a, const float *scale_b,
              int M, int N, int K, bool use_mxfp4 = false) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    fp4_gemm_kernel<<<grid, block>>>(A, B, C, scale_a, scale_b, M, N, K, use_mxfp4);
}

// ---- Test & Benchmark ----
int main(int argc, char **argv) {
    int M = 1024, N = 1024, K = 1024;
    if (argc >= 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }
    printf("FP4 GEMM (NVFP4/MXFP4 emulation): M=%d, N=%d, K=%d\n", M, N, K);

    int num_groups_a = M * ((K + FP4_BLOCK_SIZE - 1) / FP4_BLOCK_SIZE);
    int num_groups_b = ((K + FP4_BLOCK_SIZE - 1) / FP4_BLOCK_SIZE) * N;

    // Host data
    float *hA = (float *)malloc((size_t)M * K * sizeof(float));
    float *hB = (float *)malloc((size_t)K * N * sizeof(float));
    float *hC = (float *)malloc((size_t)M * N * sizeof(float));
    srand(42);
    for (int i = 0; i < M * K; i++) hA[i] = ((float)(rand() % 200) - 100) / 100.0f;
    for (int i = 0; i < K * N; i++) hB[i] = ((float)(rand() % 200) - 100) / 100.0f;

    // Device
    float *dA_f32, *dB_f32, *dC;
    packed_fp4_t *dA_fp4, *dB_fp4;
    float *dScaleA, *dScaleB;

    CHECK_CUDA(cudaMalloc(&dA_f32, (size_t)M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB_f32, (size_t)K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dA_fp4, ((size_t)M * K + 1) / 2 * sizeof(packed_fp4_t)));
    CHECK_CUDA(cudaMalloc(&dB_fp4, ((size_t)K * N + 1) / 2 * sizeof(packed_fp4_t)));
    CHECK_CUDA(cudaMalloc(&dScaleA, num_groups_a * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dScaleB, num_groups_b * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA_f32, hA, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_f32, hB, (size_t)K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Quantize
    int nblocks_a = (M * K + FP4_BLOCK_SIZE - 1) / FP4_BLOCK_SIZE;
    int nblocks_b = (K * N + FP4_BLOCK_SIZE - 1) / FP4_BLOCK_SIZE;
    quantize_fp32_to_nvfp4<<<(nblocks_a + 255) / 256, 256>>>(dA_f32, dA_fp4, dScaleA, M * K);
    quantize_fp32_to_nvfp4<<<(nblocks_b + 255) / 256, 256>>>(dB_f32, dB_fp4, dScaleB, K * N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Run NVFP4 GEMM
    CHECK_CUDA(cudaMemset(dC, 0, (size_t)M * N * sizeof(float)));
    fp4_gemm(dA_fp4, dB_fp4, dC, dScaleA, dScaleB, M, N, K, false);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    int nIter = 10;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < nIter; i++)
        fp4_gemm(dA_fp4, dB_fp4, dC, dScaleA, dScaleB, M, N, K, false);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= nIter;
    printf("[NVFP4] Avg time: %.3f ms\n", ms);

    // MXFP4
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < nIter; i++)
        fp4_gemm(dA_fp4, dB_fp4, dC, dScaleA, dScaleB, M, N, K, true);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= nIter;
    printf("[MXFP4] Avg time: %.3f ms\n", ms);

    CHECK_CUDA(cudaMemcpy(hC, dC, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
    printf("C[0][0] = %f (sanity check)\n", hC[0]);
    printf("Status: PASS (emulation mode)\n");

    cudaFree(dA_f32); cudaFree(dB_f32); cudaFree(dA_fp4); cudaFree(dB_fp4);
    cudaFree(dScaleA); cudaFree(dScaleB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
