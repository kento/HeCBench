// OpenMP target port of ludb-kokkos: batched LU decomposition.
// Column-major storage: mat[j*N + i] = element at row i, column j.

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>

#define N          48
#define BATCH_SIZE 10000

static void getLUdecoded(const float* mat, float* L, float* U) {
    memset(L, 0, N * N * sizeof(float));
    memset(U, 0, N * N * sizeof(float));
    for (int i = 0; i < N; i++) L[i * N + i] = 1.0f;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < i; j++) L[j * N + i] = mat[j * N + i];
    for (int i = 0; i < N; i++)
        for (int j = i; j < N; j++) U[j * N + i] = mat[j * N + i];
}

static void matrixMultiply(float* res, const float* mat1, const float* mat2) {
    memset(res, 0, N * N * sizeof(float));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                res[j * N + i] += mat1[k * N + i] * mat2[j * N + k];
}

int main(int argc, char* argv[]) {
    const int repeat = (argc > 1) ? atoi(argv[1]) : 1;

    float* orig_mats = new float[(size_t)BATCH_SIZE * N * N];
    srand(42);
    for (int b = 0; b < BATCH_SIZE; b++) {
        float* mat = orig_mats + (size_t)b * N * N;
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++)
                mat[j * N + i] = (float)rand() / RAND_MAX;
        for (int i = 0; i < N; i++) mat[i * N + i] += N;
    }

    float* d_mats   = (float*)malloc((size_t)BATCH_SIZE * N * N * sizeof(float));
    int*   d_pivots = (int*)  malloc((size_t)BATCH_SIZE * N * sizeof(int));

    #pragma omp target enter data \
        map(alloc: d_mats[0:BATCH_SIZE*N*N], d_pivots[0:BATCH_SIZE*N])

    double total_time = 0.0;

    for (int r = 0; r < repeat; r++) {
        memcpy(d_mats, orig_mats, (size_t)BATCH_SIZE * N * N * sizeof(float));
        #pragma omp target update to(d_mats[0:BATCH_SIZE*N*N])

        auto tstart = std::chrono::steady_clock::now();

        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int b = 0; b < BATCH_SIZE; b++) {
            const int base = b * N * N;
            const int pb   = b * N;

            for (int k = 0; k < N; k++) {
                int   piv    = k;
                float maxval = fabsf(d_mats[base + k * N + k]);
                for (int i = k + 1; i < N; i++) {
                    float v = fabsf(d_mats[base + k * N + i]);
                    if (v > maxval) { maxval = v; piv = i; }
                }
                d_pivots[pb + k] = piv + 1;

                if (piv != k) {
                    for (int j = 0; j < N; j++) {
                        float tmp                  = d_mats[base + j * N + k];
                        d_mats[base + j * N + k]   = d_mats[base + j * N + piv];
                        d_mats[base + j * N + piv] = tmp;
                    }
                }

                const float diag = d_mats[base + k * N + k];
                if (diag != 0.0f) {
                    const float inv = 1.0f / diag;
                    for (int i = k + 1; i < N; i++)
                        d_mats[base + k * N + i] *= inv;
                    for (int j = k + 1; j < N; j++)
                        for (int i = k + 1; i < N; i++)
                            d_mats[base + j * N + i] -=
                                d_mats[base + j * N + k] * d_mats[base + k * N + i];
                }
            }
        }

        auto tstop = std::chrono::steady_clock::now();
        total_time += std::chrono::duration<double>(tstop - tstart).count();
    }

    const double avg_ms = total_time * 1000.0 / repeat;
    const double gflops = (double)BATCH_SIZE * (2.0 / 3.0 * N * N * N) * repeat
                          / total_time * 1e-9;
    printf("LU batch (%d x %dx%d), repeat=%d\n", BATCH_SIZE, N, N, repeat);
    printf("Average kernel time : %.3f ms\n", avg_ms);
    printf("Throughput          : %.2f GFlops\n", gflops);

    #pragma omp target update from(d_mats[0:BATCH_SIZE*N*N], d_pivots[0:BATCH_SIZE*N])

    {
        const float* lu_mat = d_mats;
        const int*   pivot  = d_pivots;
        const float* A      = orig_mats;

        float L[N * N], U[N * N], LU[N * N], PA[N * N];
        getLUdecoded(lu_mat, L, U);
        matrixMultiply(LU, L, U);

        memcpy(PA, A, N * N * sizeof(float));
        for (int k = 0; k < N; k++) {
            const int pk = pivot[k] - 1;
            if (pk != k) {
                for (int j = 0; j < N; j++) {
                    float tmp      = PA[j * N + k];
                    PA[j * N + k]  = PA[j * N + pk];
                    PA[j * N + pk] = tmp;
                }
            }
        }

        float maxerr = 0.0f;
        for (int idx = 0; idx < N * N; idx++)
            maxerr = fmaxf(maxerr, fabsf(LU[idx] - PA[idx]));
        printf("Verification (matrix 0): max|LU - PA| = %e  %s\n",
               maxerr, maxerr < 1e-3f ? "PASS" : "FAIL");
    }

    #pragma omp target exit data \
        map(delete: d_mats[0:BATCH_SIZE*N*N], d_pivots[0:BATCH_SIZE*N])

    free(d_mats); free(d_pivots);
    delete[] orig_mats;
    return 0;
}
