/*
HPL (High Performance Linpack) simplified single-node version.
OpenMP target offloading port.

Solves Ax = b using LU factorization with partial pivoting (DGETRF).

Usage: ./main [n]   (default n=1024)
*/

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>

int main(int argc, char* argv[])
{
    int n = 1024;
    if (argc > 1) n = atoi(argv[1]);

    printf("Matrix size: %d\n", n);

    std::vector<double> A_init(n * n);
    std::vector<double> b_init(n);
    std::vector<double> x_ref (n, 1.0);

    srand(12345);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            A_init[i*n + j] = (double)rand() / RAND_MAX + (i == j ? n : 0.0);
    }
    for (int i = 0; i < n; i++) {
        double s = 0.0;
        for (int j = 0; j < n; j++) s += A_init[i*n + j];
        b_init[i] = s;
    }

    // Flat row-major A on device: d_A[i*n+j]
    double* d_A  = (double*)malloc((size_t)n * n * sizeof(double));
    double* d_b  = (double*)malloc(n * sizeof(double));
    int*    d_piv = (int*)malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) d_A[i*n+j] = A_init[i*n+j];
        d_b[i] = b_init[i];
    }

    #pragma omp target enter data map(to: d_A[0:n*n], d_b[0:n]) \
        map(alloc: d_piv[0:n])

    auto t_start = std::chrono::high_resolution_clock::now();

    // LU factorization with partial pivoting
    for (int k = 0; k < n; k++) {
        // Find pivot row
        double max_val = 0.0;
        #pragma omp target teams distribute parallel for reduction(max:max_val) thread_limit(256)
        for (int i = k; i < n; i++) {
            double av = d_A[i*n + k];
            if (av < 0.0) av = -av;
            if (av > max_val) max_val = av;
        }

        int pivot_row = k;
        const double pv = max_val;
        #pragma omp target teams distribute parallel for reduction(min:pivot_row) thread_limit(256)
        for (int i = k; i < n; i++) {
            double av = d_A[i*n + k];
            if (av < 0.0) av = -av;
            if (av == pv && i < pivot_row) pivot_row = i;
        }

        const int pr = pivot_row;
        const int kk = k;
        #pragma omp target
        { d_piv[kk] = pr; }

        // Swap rows k and pivot_row
        if (pivot_row != k) {
            #pragma omp target teams distribute parallel for thread_limit(256)
            for (int j = 0; j < n; j++) {
                double tmp    = d_A[kk*n + j];
                d_A[kk*n + j]  = d_A[pr*n + j];
                d_A[pr*n + j]  = tmp;
            }
            #pragma omp target
            { double tmp = d_b[kk]; d_b[kk] = d_b[pr]; d_b[pr] = tmp; }
        }

        // Scale column k below pivot
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int i = k+1; i < n; i++) {
            d_A[i*n + kk] /= d_A[kk*n + kk];
        }

        // Update trailing submatrix
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int i = k+1; i < n; i++) {
            double aik = d_A[i*n + kk];
            for (int j = kk + 1; j < n; j++)
                d_A[i*n + j] -= aik * d_A[kk*n + j];
        }
    }

    // Forward substitution: Ly = b
    double* d_y = (double*)malloc(n * sizeof(double));
    #pragma omp target update from(d_b[0:n])
    for (int i = 0; i < n; i++) d_y[i] = d_b[i];
    #pragma omp target enter data map(to: d_y[0:n])

    for (int i = 0; i < n; i++) {
        const int ii = i;
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int j = i+1; j < n; j++) {
            d_y[j] -= d_A[j*n + ii] * d_y[ii];
        }
    }

    // Backward substitution: Ux = y
    double* d_x = (double*)malloc(n * sizeof(double));
    #pragma omp target update from(d_y[0:n])
    for (int i = 0; i < n; i++) d_x[i] = d_y[i];
    #pragma omp target enter data map(to: d_x[0:n])

    for (int i = n-1; i >= 0; i--) {
        const int ii = i;
        #pragma omp target
        { d_x[ii] /= d_A[ii*n + ii]; }
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int j = 0; j < i; j++) {
            d_x[j] -= d_A[j*n + ii] * d_x[ii];
        }
    }

    #pragma omp target update from(d_x[0:n])
    auto t_end = std::chrono::high_resolution_clock::now();
    double lu_time = std::chrono::duration<double>(t_end - t_start).count();
    printf("LU factorization time: %.6f (s)\n", lu_time);

    // Verify residual
    double norm_r = 0.0, norm_A = 0.0, norm_x = 0.0;
    for (int i = 0; i < n; i++) {
        double ax_i = 0.0;
        for (int j = 0; j < n; j++) ax_i += A_init[i*n + j] * d_x[j];
        double ri = ax_i - b_init[i];
        norm_r += ri * ri;
    }
    norm_r = sqrt(norm_r);
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < n; j++) row_sum += fabs(A_init[i*n + j]);
        norm_A = std::max(norm_A, row_sum);
        norm_x += d_x[i] * d_x[i];
    }
    norm_x = sqrt(norm_x);
    double residual = norm_r / (norm_A * norm_x * n);
    printf("Residual ||Ax-b||/||A||/||x||/N: %e\n", residual);
    printf("%s\n", residual < 1e-8 ? "PASSED" : "FAILED");

    #pragma omp target exit data map(delete: d_A[0:n*n], d_b[0:n], d_piv[0:n], d_y[0:n], d_x[0:n])
    free(d_A); free(d_b); free(d_piv); free(d_y); free(d_x);
    return 0;
}
