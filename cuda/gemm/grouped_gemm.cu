/**
 * Grouped GEMM - Batched matrix multiplication for LoRA and multi-expert routing
 * Computes: C_i = A_i @ B_i for i = 0..num_groups-1
 * Each group can have different M dimension but shares K and N.
 *
 * Use cases:
 *   - LoRA: base_output + sum(A_i @ B_i) for multiple LoRA adapters
 *   - MoE: per-expert linear transforms routed to different subsets
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

// Problem descriptor for each group
struct GemmProblem {
    int M, N, K;
    int A_offset;  // Byte offset into A buffer
    int B_offset;  // Byte offset into B buffer
    int C_offset;  // Byte offset into C buffer
};

// ========== Fixed-size Grouped GEMM ==========
// All groups have the same M, N, K (batched GEMM)
__global__ void grouped_gemm_fixed_kernel(
    const float *__restrict__ A,  // [num_groups, M, K]
    const float *__restrict__ B,  // [num_groups, K, N]
    float *__restrict__ C,        // [num_groups, M, N]
    int M, int N, int K, int num_groups) {

    const int TILE = 16;
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int group = blockIdx.z;
    if (group >= num_groups) return;

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    size_t a_base = (size_t)group * M * K;
    size_t b_base = (size_t)group * K * N;
    size_t c_base = (size_t)group * M * N;

    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K)
            ? A[a_base + row * K + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < K && col < N)
            ? B[b_base + bRow * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++)
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N)
        C[c_base + row * N + col] = acc;
}

// ========== Variable-size Grouped GEMM ==========
// Each group can have different dimensions - uses problem descriptors
__global__ void grouped_gemm_variable_kernel(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C,
    const GemmProblem *__restrict__ problems,
    int num_groups) {

    int group = blockIdx.z;
    if (group >= num_groups) return;

    GemmProblem prob = problems[group];
    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;

    if (row >= prob.M || col >= prob.N) return;

    const float *a_ptr = A + prob.A_offset;
    const float *b_ptr = B + prob.B_offset;
    float *c_ptr = C + prob.C_offset;

    float acc = 0.0f;
    for (int k = 0; k < prob.K; k++) {
        acc += a_ptr[row * prob.K + k] * b_ptr[k * prob.N + col];
    }
    c_ptr[row * prob.N + col] = acc;
}

// ========== LoRA Fused GEMM: Y = X @ W + sum(X @ A_i @ B_i * scale_i) ==========
__global__ void lora_fused_gemm_kernel(
    const float *__restrict__ X,      // [M, K_in]
    const float *__restrict__ W,      // [K_in, N]
    const float *__restrict__ loraA,  // [num_loras, K_in, R]
    const float *__restrict__ loraB,  // [num_loras, R, N]
    const float *__restrict__ scales, // [num_loras]
    float *__restrict__ Y,            // [M, N]
    int M, int N, int K_in, int R, int num_loras) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    // Base: X @ W
    float val = 0.0f;
    for (int k = 0; k < K_in; k++)
        val += X[row * K_in + k] * W[k * N + col];

    // LoRA sum: sum_i (scale_i * X @ A_i @ B_i)
    for (int l = 0; l < num_loras; l++) {
        const float *A_l = loraA + (size_t)l * K_in * R;
        const float *B_l = loraB + (size_t)l * R * N;
        float lora_val = 0.0f;
        for (int r = 0; r < R; r++) {
            float xa = 0.0f;
            for (int k = 0; k < K_in; k++)
                xa += X[row * K_in + k] * A_l[k * R + r];
            lora_val += xa * B_l[r * N + col];
        }
        val += scales[l] * lora_val;
    }

    Y[row * N + col] = val;
}

// Host wrappers
void grouped_gemm_fixed(const float *A, const float *B, float *C,
                         int M, int N, int K, int num_groups) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16, num_groups);
    grouped_gemm_fixed_kernel<<<grid, block>>>(A, B, C, M, N, K, num_groups);
}

void grouped_gemm_variable(const float *A, const float *B, float *C,
                            const GemmProblem *problems, int num_groups,
                            int max_M, int max_N) {
    dim3 block(16, 16);
    dim3 grid((max_N + 15) / 16, (max_M + 15) / 16, num_groups);
    grouped_gemm_variable_kernel<<<grid, block>>>(A, B, C, problems, num_groups);
}

// ---- Test & Benchmark ----
int main(int argc, char **argv) {
    int num_groups = 8, M = 512, N = 512, K = 256;
    if (argc >= 5) {
        num_groups = atoi(argv[1]); M = atoi(argv[2]);
        N = atoi(argv[3]); K = atoi(argv[4]);
    }
    printf("Grouped GEMM: groups=%d, M=%d, N=%d, K=%d\n", num_groups, M, N, K);

    size_t sizeA = (size_t)num_groups * M * K * sizeof(float);
    size_t sizeB = (size_t)num_groups * K * N * sizeof(float);
    size_t sizeC = (size_t)num_groups * M * N * sizeof(float);

    float *hA = (float *)malloc(sizeA);
    float *hB = (float *)malloc(sizeB);
    float *hC = (float *)malloc(sizeC);

    srand(42);
    for (size_t i = 0; i < (size_t)num_groups * M * K; i++)
        hA[i] = ((float)(rand() % 200) - 100) / 100.0f;
    for (size_t i = 0; i < (size_t)num_groups * K * N; i++)
        hB[i] = ((float)(rand() % 200) - 100) / 100.0f;

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, sizeA));
    CHECK_CUDA(cudaMalloc(&dB, sizeB));
    CHECK_CUDA(cudaMalloc(&dC, sizeC));
    CHECK_CUDA(cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice));

    // Warmup + run
    grouped_gemm_fixed(dA, dB, dC, M, N, K, num_groups);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    int nIter = 20;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < nIter; i++)
        grouped_gemm_fixed(dA, dB, dC, M, N, K, num_groups);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= nIter;

    double total_flops = 2.0 * num_groups * M * N * K;
    double tflops = total_flops / (ms * 1e-3) / 1e12;
    printf("Avg time: %.3f ms | %.2f TFLOPS (aggregate)\n", ms, tflops);

    // Correctness check (first group, small block)
    CHECK_CUDA(cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost));
    int check = min(M, 32);
    float maxErr = 0;
    for (int i = 0; i < check; i++) {
        for (int j = 0; j < check; j++) {
            float ref = 0;
            for (int k = 0; k < K; k++)
                ref += hA[i * K + k] * hB[k * N + j];
            float err = fabsf(hC[i * N + j] - ref);
            if (err > maxErr) maxErr = err;
        }
    }
    printf("Max error (group 0, %dx%d): %e -> %s\n", check, check, maxErr,
           maxErr < 1e-3f ? "PASS" : "FAIL");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
