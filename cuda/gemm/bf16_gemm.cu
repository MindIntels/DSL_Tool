/**
 * BF16 GEMM using Tensor Cores (WMMA API)
 * C = alpha * A @ B + beta * C
 * A: [M, K] BF16, B: [K, N] BF16, C: [M, N] FP32
 * Requires SM80+ (Ampere and above)
 *
 * Design: each warp owns exactly one 16×16 WMMA output tile.
 * Block covers (WARPS_M × WARPS_N) output tiles with shared-memory staging.
 */
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Number of WMMA tiles per block in M and N dimensions
#define WARPS_M 4
#define WARPS_N 4

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// Shared-memory staged BF16 GEMM using WMMA
// Grid: (ceil(M/(WARPS_M*16)), ceil(N/(WARPS_N*16)))
// Block: WARPS_M * WARPS_N * 32 threads
__global__ void bf16_gemm_kernel(const __nv_bfloat16 *__restrict__ A,
                                  const __nv_bfloat16 *__restrict__ B,
                                  float *__restrict__ C,
                                  int M, int N, int K,
                                  float alpha, float beta) {
    // Shared memory staging for one K-slice
    // sA: [WARPS_M * 16][WMMA_K]  sB: [WMMA_K][WARPS_N * 16]
    // Pad columns to avoid bank conflicts on 128-byte cache lines
    __shared__ __nv_bfloat16 sA[WARPS_M * WMMA_M][WMMA_K + 8];
    __shared__ __nv_bfloat16 sB[WMMA_K][WARPS_N * WMMA_N + 8];

    const int warpId  = threadIdx.x / 32;
    const int laneId  = threadIdx.x % 32;

    // Each warp maps to a (warpRow, warpCol) output tile
    const int warpRow = warpId / WARPS_N;   // [0, WARPS_M)
    const int warpCol = warpId % WARPS_N;   // [0, WARPS_N)

    // Top-left corner of this block's output region
    const int blockRowStart = blockIdx.x * (WARPS_M * WMMA_M);
    const int blockColStart = blockIdx.y * (WARPS_N * WMMA_N);

    // WMMA fragments
    wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>                           c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K in WMMA_K steps, staging via shared memory
    for (int k0 = 0; k0 < K; k0 += WMMA_K) {

        // ---- Cooperative load of sA [WARPS_M*16, WMMA_K] ----
        // Total elements = WARPS_M * 16 * WMMA_K = 4*16*16 = 1024
        // blockDim.x = WARPS_M * WARPS_N * 32 = 512
        // Each thread loads 2 elements
        {
            int total = WARPS_M * WMMA_M * WMMA_K;
            for (int i = threadIdx.x; i < total; i += blockDim.x) {
                int r = i / WMMA_K;
                int c = i % WMMA_K;
                int gRow = blockRowStart + r;
                int gCol = k0 + c;
                sA[r][c] = (gRow < M && gCol < K)
                               ? A[gRow * K + gCol]
                               : __float2bfloat16(0.0f);
            }
        }

        // ---- Cooperative load of sB [WMMA_K, WARPS_N*16] ----
        {
            int total = WMMA_K * WARPS_N * WMMA_N;
            for (int i = threadIdx.x; i < total; i += blockDim.x) {
                int r = i / (WARPS_N * WMMA_N);
                int c = i % (WARPS_N * WMMA_N);
                int gRow = k0 + r;
                int gCol = blockColStart + c;
                sB[r][c] = (gRow < K && gCol < N)
                               ? B[gRow * N + gCol]
                               : __float2bfloat16(0.0f);
            }
        }

        __syncthreads();

        // ---- Each warp loads its fragment and accumulates ----
        // A fragment: sA[warpRow*16 .. warpRow*16+15][0..15], stride = WMMA_K+8
        wmma::load_matrix_sync(a_frag, &sA[warpRow * WMMA_M][0], WMMA_K + 8);
        // B fragment: sB[0..15][warpCol*16 .. warpCol*16+15], stride = WARPS_N*16+8
        wmma::load_matrix_sync(b_frag, &sB[0][warpCol * WMMA_N], WARPS_N * WMMA_N + 8);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // ---- Scale and store ----
    if (alpha != 1.0f) {
        for (int i = 0; i < c_frag.num_elements; i++)
            c_frag.x[i] *= alpha;
    }

    int cRow = blockRowStart + warpRow * WMMA_M;
    int cCol = blockColStart + warpCol * WMMA_N;

    if (cRow < M && cCol < N) {
        if (beta != 0.0f) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_old;
            wmma::load_matrix_sync(c_old, C + cRow * N + cCol, N, wmma::mem_row_major);
            for (int i = 0; i < c_frag.num_elements; i++)
                c_frag.x[i] += beta * c_old.x[i];
        }
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

// Host wrapper
void bf16_gemm(const __nv_bfloat16 *A, const __nv_bfloat16 *B, float *C,
               int M, int N, int K, float alpha = 1.0f, float beta = 0.0f) {
    dim3 block(WARPS_M * WARPS_N * 32);  // 512 threads
    dim3 grid((M + WARPS_M * WMMA_M - 1) / (WARPS_M * WMMA_M),
              (N + WARPS_N * WMMA_N - 1) / (WARPS_N * WMMA_N));
    bf16_gemm_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
}

// ---- Benchmark & Test ----
double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

int main(int argc, char **argv) {
    int M = 4096, N = 4096, K = 4096;
    if (argc >= 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }

    printf("BF16 GEMM (WMMA): M=%d, N=%d, K=%d\n", M, N, K);

    size_t sizeA = (size_t)M * K * sizeof(__nv_bfloat16);
    size_t sizeB = (size_t)K * N * sizeof(__nv_bfloat16);
    size_t sizeC = (size_t)M * N * sizeof(float);

    // Allocate host
    __nv_bfloat16 *hA = (__nv_bfloat16 *)malloc(sizeA);
    __nv_bfloat16 *hB = (__nv_bfloat16 *)malloc(sizeB);
    float *hC = (float *)malloc(sizeC);
    float *hC_ref = (float *)malloc(sizeC);

    // Init random
    srand(42);
    for (int i = 0; i < M * K; i++) hA[i] = __float2bfloat16((float)(rand() % 10 - 5) / 5.0f);
    for (int i = 0; i < K * N; i++) hB[i] = __float2bfloat16((float)(rand() % 10 - 5) / 5.0f);
    memset(hC, 0, sizeC);

    // Allocate device
    __nv_bfloat16 *dA, *dB;
    float *dC;
    CHECK_CUDA(cudaMalloc(&dA, sizeA));
    CHECK_CUDA(cudaMalloc(&dB, sizeB));
    CHECK_CUDA(cudaMalloc(&dC, sizeC));

    CHECK_CUDA(cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, sizeC));

    // Warmup
    bf16_gemm(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    int nIter = 20;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < nIter; i++) {
        bf16_gemm(dA, dB, dC, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= nIter;

    double tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12;
    printf("Avg time: %.3f ms | %.2f TFLOPS\n", ms, tflops);

    // Correctness check (small sub-block)
    CHECK_CUDA(cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost));
    int checkSize = (M < 64) ? M : 64;
    float maxErr = 0;
    for (int i = 0; i < checkSize; i++) {
        for (int j = 0; j < checkSize; j++) {
            float ref = 0;
            for (int k = 0; k < K; k++) {
                ref += __bfloat162float(hA[i * K + k]) * __bfloat162float(hB[k * N + j]);
            }
            float err = fabsf(hC[i * N + j] - ref);
            if (err > maxErr) maxErr = err;
        }
    }
    printf("Max error (first %dx%d block): %e\n", checkSize, checkSize, maxErr);
    printf("Status: %s\n", maxErr < 1.0f ? "PASS" : "FAIL");

    // Cleanup
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC); free(hC_ref);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
