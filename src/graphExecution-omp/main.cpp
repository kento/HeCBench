// OpenMP target offloading port of graphExecution-kokkos

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>

#define LAUNCH_ITERATIONS 3

static void init_input(float *a, size_t size)
{
  srand(123);
  for (size_t i = 0; i < size; i++)
    a[i] = (rand() & 0xFF) / (float)RAND_MAX;
}

static void run_reduce(const char *label,
                       float* d_input,
                       float* h_input,
                       size_t size)
{
  for (int iter = 0; iter < LAUNCH_ITERATIONS; iter++) {
    // Copy input to device
    for (size_t i = 0; i < size; i++) d_input[i] = h_input[i];
    #pragma omp target update to(d_input[0:size])

    auto start = std::chrono::steady_clock::now();

    double result = 0.0;
    for (int i = 0; i < 100; i++) {
      double partial = 0.0;
      #pragma omp target teams distribute parallel for reduction(+:partial) thread_limit(256)
      for (size_t j = 0; j < size; j++) {
        partial += (double)d_input[j];
      }
      result = partial;
    }

    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  end - start).count();
    printf("[%s] final reduced sum = %lf\n", label, result);
    printf("Execution time: %f (us)\n\n", time * 1e-3f);
  }
}

int main(int argc, char **argv)
{
  for (size_t size = 512; size <= (size_t)(1 << 27); size *= 512) {
    printf("\n-----------------------------\n");
    printf("%zu elements\n", size);
    printf("Launch iterations = %d\n", LAUNCH_ITERATIONS);

    float* d_input = (float*)malloc(size * sizeof(float));
    float* h_input = (float*)malloc(size * sizeof(float));
    init_input(h_input, size);

    #pragma omp target enter data map(alloc: d_input[0:size])

    run_reduce("usingGraph",  d_input, h_input, size);
    run_reduce("UsingStream", d_input, h_input, size);

    #pragma omp target exit data map(delete: d_input[0:size])
    free(d_input);
    free(h_input);
  }
  return 0;
}
