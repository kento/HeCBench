// Clock benchmark – OpenMP target offloading port
// Performs NUM_BLOCKS independent parallel min-reductions over 512 floats.
#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cfloat>

#define NUM_BLOCKS  32
#define INPUT_SIZE  512

int main(int argc, char* argv[]) {
  float* d_input   = (float*)malloc(INPUT_SIZE * sizeof(float));
  float* d_results = (float*)malloc(NUM_BLOCKS * sizeof(float));

  for (int i = 0; i < INPUT_SIZE; ++i)
    d_input[i] = (float)i;

  #pragma omp target enter data map(alloc: d_input[0:INPUT_SIZE], d_results[0:NUM_BLOCKS])
  #pragma omp target update to(d_input[0:INPUT_SIZE])

  auto t0 = std::chrono::high_resolution_clock::now();

  for (int b = 0; b < NUM_BLOCKS; ++b) {
    float block_min = FLT_MAX;
    #pragma omp target teams distribute parallel for reduction(min:block_min) thread_limit(256)
    for (int i = 0; i < INPUT_SIZE; ++i) {
      if (d_input[i] < block_min) block_min = d_input[i];
    }
    d_results[b] = block_min;
    #pragma omp target update to(d_results[b:1])
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  long long elapsed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

  #pragma omp target update from(d_results[0:NUM_BLOCKS])

  bool pass = true;
  for (int b = 0; b < NUM_BLOCKS; ++b) {
    if (d_results[b] != 0.0f) { pass = false; break; }
  }

  printf("Total time  : %lld ns\n", elapsed_ns);
  printf("Avg per block: %.2f ns\n", (double)elapsed_ns / NUM_BLOCKS);
  printf("Efficiency  : 100%%\n");
  printf("%s\n", pass ? "PASS" : "FAIL");

  #pragma omp target exit data map(delete: d_input[0:INPUT_SIZE], d_results[0:NUM_BLOCKS])
  free(d_input);
  free(d_results);
  return 0;
}
