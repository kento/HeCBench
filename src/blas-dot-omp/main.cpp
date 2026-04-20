/*
 * OpenMP target offloading port of blas-dot.
 */

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <vector>
#include <omp.h>

template <typename T>
void dot(const size_t iNumElements, const int iNumIterations) {
  std::vector<T> h_A(iNumElements), h_B(iNumElements);

  double sum = 0.0;
  double val = sqrt(65504.0 / iNumElements);
  for (size_t i = 0; i < iNumElements; i++) {
    h_A[i] = (T)val;
    h_B[i] = (T)val;
    sum += (double)h_A[i] * (double)h_B[i];
  }

  T* d_A = (T*)malloc(iNumElements * sizeof(T));
  T* d_B = (T*)malloc(iNumElements * sizeof(T));
  for (size_t i = 0; i < iNumElements; i++) { d_A[i] = h_A[i]; d_B[i] = h_B[i]; }

  #pragma omp target enter data map(to: d_A[0:iNumElements], d_B[0:iNumElements])

  auto start = std::chrono::steady_clock::now();

  double device_dot = 0.0;
  for (int i = 0; i < iNumIterations; i++) {
    double local_sum = 0.0;
    #pragma omp target teams distribute parallel for reduction(+:local_sum) thread_limit(256)
    for (size_t j = 0; j < iNumElements; j++) {
      local_sum += (double)d_A[j] * (double)d_B[j];
    }
    device_dot = local_sum;
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average dot execution time %f (ms)\n", (time * 1e-6f) / iNumIterations);
  printf("Host: %lf  Device: %lf\n", sum, device_dot);
  printf("%s\n\n", (fabs(device_dot - sum) < 1e-1) ? "PASS" : "FAIL");

  #pragma omp target exit data map(delete: d_A[0:iNumElements], d_B[0:iNumElements])
  free(d_A);
  free(d_B);
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const size_t iNumElements    = atol(argv[1]);
  const int    iNumIterations  = atoi(argv[2]);

  printf("\nFP64 Dot\n");
  dot<double>(iNumElements, iNumIterations);
  printf("\nFP32 Dot\n");
  dot<float>(iNumElements, iNumIterations);
  return 0;
}
