#include "hip/hip_runtime.h"
#include "hip/hip_runtime.h"
#include <algorithm>
#include <chrono> // for high_resolution_clock
#include <cstdio>
#include <random>

#include "reference.h"
#include "gpu_solver.h"

void generate_data(int size, int min, int max, float *data) {
  std::mt19937_64 generator{1993764};
  std::uniform_int_distribution<> dist{min, max};
  for (int i = 0; i < size; ++i) {
    data[i] = dist(generator);
  }
}


int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  int repeat = atoi(argv[1]);

  // number of functions
  int N = 1999999;
  printf("N = %d\n", N);

  float *A, *B, *C, *D, *E;
  float *minimum_ref, *minimum;

  checkCuda(hipHostMalloc((void **)&A, N * sizeof(float)));
  checkCuda(hipHostMalloc((void **)&B, N * sizeof(float)));
  checkCuda(hipHostMalloc((void **)&C, N * sizeof(float)));
  checkCuda(hipHostMalloc((void **)&D, N * sizeof(float)));
  checkCuda(hipHostMalloc((void **)&E, N * sizeof(float)));
  checkCuda(hipHostMalloc((void **)&minimum_ref, N * sizeof(float)));
  checkCuda(hipHostMalloc((void **)&minimum, N * sizeof(float)));

  printf("generating data...\n");

  generate_data(N, -100, 100, A);
  generate_data(N, -100, 100, B);
  generate_data(N, -100, 100, C);
  generate_data(N, -100, 100, D);
  generate_data(N, -100, 100, E);

  for (int i = 0; i < N; i++) {
    if (A[i] == 0) {
      A[i] = 1;
    } // avoid undefined behaviour in solver when A=0
  }

  float dur = 0;
  float avg = 0;
  bool ok;

  printf("####################### Reference #############\n");

  for (int k = 0; k < repeat; ++k) {
    auto start = std::chrono::high_resolution_clock::now();

    QuarticMinimumCPU(N, A, B, C, D, E, minimum_ref);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = end - start;
    dur = elapsed.count() * 1000;
    // printf("Time (ms): %f\n", dur);
    avg += dur;
  }

  printf("Execution time (ms): %f\n", avg / repeat);

  avg = 0;

  printf("####################### GPU (no streams) #############\n");

  for (int k = 0; k < repeat; ++k) {

    auto start = std::chrono::high_resolution_clock::now();

    QuarticMinimumGPU(N, A, B, C, D, E, minimum);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = end - start;
    dur = elapsed.count() * 1000;
    // printf("Time (ms): %f\n", dur);
    avg += dur;
  }

  printf("Execution time (ms): %f\n", avg / repeat);

  ok = true;
  for (int i = 0; i < N; i++) {
    if (fabsf(minimum[i] - minimum_ref[i]) > 1e-3f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  avg = 0;

  printf("####################### GPU (streams) #############\n");

  for (int k = 0; k < repeat; ++k) {

    auto start = std::chrono::high_resolution_clock::now();

    QuarticMinimumGPUStreams(N, A, B, C, D, E, minimum);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = end - start;
    dur = elapsed.count() * 1000;
    // printf("Time (ms): %f\n", dur);
    avg += dur;
  }

  printf("Execution time (ms): %f\n", avg / repeat);

  ok = true;
  for (int i = 0; i < N; i++) {
    if (fabsf(minimum[i] - minimum_ref[i]) > 1e-3f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  checkCuda(hipHostFree(A));
  checkCuda(hipHostFree(B));
  checkCuda(hipHostFree(C));
  checkCuda(hipHostFree(D));
  checkCuda(hipHostFree(E));
  checkCuda(hipHostFree(minimum_ref));
  checkCuda(hipHostFree(minimum));

  return 0;
}
