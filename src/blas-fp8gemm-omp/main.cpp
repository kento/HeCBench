/*
 * OpenMP target offloading port of blas-fp8gemm.
 * Implements FP32 GEMM on device (simulating FP8 range).
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int m = 64, n = 128, k = 256;
  const float alpha = 2.0f, beta = 1.0f;

  float* A = (float*)malloc(k * m * sizeof(float));
  float* B = (float*)malloc(k * n * sizeof(float));
  float* C = (float*)malloc(m * n * sizeof(float));
  float* D = (float*)malloc(m * n * sizeof(float));

  for (int i = 0; i < k * m; i++) A[i] = (float)(i % 9 - 4) / 4.f;
  for (int i = 0; i < k * n; i++) B[i] = (float)(i % 9 - 4) / 4.f;
  for (int i = 0; i < m * n; i++) C[i] = 0.5f;

  #pragma omp target enter data map(to: A[0:k*m], B[0:k*n], C[0:m*n]) \
                                map(alloc: D[0:m*n])

  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    // D = C
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < m * n; i++) D[i] = C[i];

    // GEMM: D = alpha * A^T * B + beta * D  (col-major)
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int idx = 0; idx < m * n; idx++) {
      int col = idx / m;
      int row = idx % m;
      float s = 0.f;
      for (int i = 0; i < k; i++)
        s += A[row + i * k] * B[i + col * k];
      D[row + col * m] = alpha * s + beta * D[row + col * m];
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average GEMM execution time %f (us)\n", (time * 1e-3f) / repeat);

  #pragma omp target update from(D[0:m*n])
  float s = 0.f;
  for (int i = 0; i < m * n; i++) s += D[i];
  printf("Checksum: %f\n", s / (m * n));

  #pragma omp target exit data map(delete: A[0:k*m], B[0:k*n], C[0:m*n], D[0:m*n])
  free(A); free(B); free(C); free(D);
  return 0;
}
