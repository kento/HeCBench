// OpenMP target offloading port of prefetch benchmark.
// B[i] += A[i] repeated `repeat` times, verifying B[i] == repeat + 2.

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

static constexpr int NUM_ELEMENTS = 64 * 1024 * 1024;

void run_benchmark(const char *label, int numElements, int repeat)
{
  printf("%s\n", label);

  float *A = (float*)malloc(numElements * sizeof(float));
  float *B = (float*)malloc(numElements * sizeof(float));

#pragma omp target enter data map(alloc: A[0:numElements], B[0:numElements])

  // Initialize: A[i]=1, B[i]=2
#pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < numElements; i++) {
    A[i] = 1.0f;
    B[i] = 2.0f;
  }

  auto start = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
#pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < numElements; i++)
      B[i] += A[i];
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time: %f (ms)\n", time * 1e-6f / repeat);

#pragma omp target update from(B[0:numElements])

  float maxError = 0.0f;
  for (int i = 0; i < numElements; i++)
    maxError = std::fmax(maxError, std::fabs(B[i] - (float)(repeat + 2)));

  printf("%s\n", (maxError == 0.0f) ? "PASS" : "FAIL");

#pragma omp target exit data map(delete: A[0:numElements], B[0:numElements])
  free(A);
  free(B);
}

int main(int argc, char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  for (int i = 0; i < 10; i++)
    run_benchmark("Concurrent managed access with prefetch", NUM_ELEMENTS, repeat);

  for (int i = 0; i < 10; i++)
    run_benchmark("Concurrent managed access without prefetch", NUM_ELEMENTS, repeat);

  return 0;
}
