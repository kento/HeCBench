/*
 * OpenMP target offloading port of allreduce benchmark.
 */

#include <chrono>
#include <cstdio>
#include <cmath>
#include <vector>
#include <omp.h>

void allreduce_sum(float* data, int n) {
  float sum = 0.f;
  #pragma omp target teams distribute parallel for reduction(+:sum) thread_limit(256)
  for (int i = 0; i < n; i++) sum += data[i];

  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < n; i++) data[i] = sum;
}

void run_test(size_t n, int iterations) {
  float* data = (float*)malloc(n * sizeof(float));
  for (size_t i = 0; i < n; i++) data[i] = 1.0f;

  #pragma omp target enter data map(to: data[0:n])

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; i++) {
    allreduce_sum(data, (int)n);
  }
  auto end = std::chrono::steady_clock::now();
  auto time_us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-3;

  #pragma omp target update from(data[0:n])
  bool ok = (fabsf(data[0] - (float)n) < 1.0f);
  printf("  size=%zu  avg=%.2f us  %s\n", n, time_us / iterations, ok ? "PASS" : "FAIL");

  #pragma omp target exit data map(delete: data[0:n])
  free(data);
}

int main(int argc, char** argv) {
  printf("OpenMP target allreduce (single-device simulation)\n");
  std::vector<size_t> sizes = {1<<20, 1<<22, 1<<24, 1<<26};
  int iterations = 10;
  for (size_t s : sizes)
    run_test(s, iterations);
  return 0;
}
