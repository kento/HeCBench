// OpenMP target offloading port of gemv-kokkos (matrix-vector multiply)

#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    unsigned int size = 512;
    unsigned int iter = 1;
    unsigned int block_dim_x = 32;
    unsigned int block_dim_y = 4;

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--size") == 0 || strcmp(argv[i], "-s") == 0) && i + 1 < argc)
            size = (unsigned int)atoi(argv[++i]);
        else if ((strcmp(argv[i], "--iter") == 0 || strcmp(argv[i], "-i") == 0) && i + 1 < argc)
            iter = (unsigned int)atoi(argv[++i]);
        else if ((strcmp(argv[i], "--block_x") == 0 || strcmp(argv[i], "-x") == 0) && i + 1 < argc)
            block_dim_x = (unsigned int)atoi(argv[++i]);
        else if ((strcmp(argv[i], "--block_y") == 0 || strcmp(argv[i], "-y") == 0) && i + 1 < argc)
            block_dim_y = (unsigned int)atoi(argv[++i]);
    }
    (void)block_dim_x;
    (void)block_dim_y;

    printf("Testing GEMV with size=%u\n", size);

    const unsigned int n = size;
    size_t mat_size = (size_t)n * n;

    float* d_M = (float*)malloc(mat_size * sizeof(float));
    float* d_x = (float*)malloc(n * sizeof(float));
    float* d_y = (float*)malloc(n * sizeof(float));

    for (size_t i = 0; i < mat_size; i++) d_M[i] = 1.0f / (float)n;
    for (unsigned int i = 0; i < n; i++) d_x[i] = 1.0f;

    #pragma omp target enter data map(to: d_M[0:mat_size], d_x[0:n]) map(alloc: d_y[0:n])

    // Warm-up
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int row = 0; row < (int)n; row++) {
        float sum = 0.0f;
        for (unsigned int j = 0; j < n; j++)
            sum += d_M[(size_t)row * n + j] * d_x[j];
        d_y[row] = sum;
    }

    auto t_start = std::chrono::steady_clock::now();
    for (unsigned int it = 0; it < iter; it++) {
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int row = 0; row < (int)n; row++) {
            float sum = 0.0f;
            for (unsigned int j = 0; j < n; j++)
                sum += d_M[(size_t)row * n + j] * d_x[j];
            d_y[row] = sum;
        }
    }
    auto t_end = std::chrono::steady_clock::now();

    double total_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          t_end - t_start).count() * 1e-6;
    printf("Kernel time: %.3f ms\n", total_ms / (double)iter);

    #pragma omp target exit data map(delete: d_M[0:mat_size], d_x[0:n], d_y[0:n])

    free(d_M); free(d_x); free(d_y);
    return 0;
}
