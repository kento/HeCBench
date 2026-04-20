// Determinant benchmark – OpenMP target offloading port
#include <stdio.h>
#include <chrono>
#include <omp.h>
#include <cmath>

static void choleskyDecomp(float* A, int n)
{
  // Sequential Cholesky on device using a single target region
  #pragma omp target
  {
    for (int j = 0; j < n; j++) {
      float s = A[j * n + j];
      for (int k = 0; k < j; k++) s -= A[j * n + k] * A[j * n + k];
      A[j * n + j] = sqrtf(s);
      for (int i = j + 1; i < n; i++) {
        float t = A[i * n + j];
        for (int k = 0; k < j; k++) t -= A[i * n + k] * A[j * n + k];
        A[i * n + j] = t / A[j * n + j];
      }
    }
  }
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int N = 11;

  float h_A[N * N] = {
     2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2.,
    -2.,  4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
    -2.,  0.,  6.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
    -2.,  0.,  2.,  8.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,
    -2.,  0.,  2.,  4.,  10., 10., 10., 10., 10., 10., 10.,
    -2.,  0.,  2.,  4.,  10., 12., 12., 12., 12., 12., 12.,
    -2.,  0.,  2.,  4.,  10., 12., 14., 14., 14., 14., 14.,
    -2.,  0.,  2.,  4.,  10., 12., 14., 16., 16., 16., 16.,
    -2.,  0.,  2.,  4.,  10., 12., 14., 16., 18., 18., 18.,
    -2.,  0.,  2.,  4.,  10., 12., 14., 16., 18., 20., 20.,
    -2.,  0.,  2.,  4.,  10., 12., 14., 16., 18., 20., 22.
  };

  float* d_A = (float*)malloc(N * N * sizeof(float));

  #pragma omp target enter data map(alloc: d_A[0:N*N])

  float det = 0.f;

  auto start = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    for (int i = 0; i < N * N; i++) d_A[i] = h_A[i];
    #pragma omp target update to(d_A[0:N*N])

    choleskyDecomp(d_A, N);

    float prod = 1.f;
    #pragma omp target teams distribute parallel for reduction(*:prod) thread_limit(256)
    for (int i = 0; i < N; i++) {
      prod *= d_A[i * N + i];
    }
    det = prod;
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time: %f (us)\n", (time * 1e-3f) / repeat);
  printf("determinant = %f\n", det * det);

  #pragma omp target exit data map(delete: d_A[0:N*N])
  free(d_A);
  return 0;
}
