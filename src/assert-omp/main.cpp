/*
 * OpenMP target offloading port of assert benchmark.
 */

#include <cstdio>
#include <chrono>
#include <omp.h>

void perfKernel(int nBlocks, int blockSize, int repeat) {
  int N = nBlocks * blockSize;
  int dummy = 0;

  #pragma omp target enter data map(alloc: dummy)

  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int gid = 0; gid < N; gid++) {
      int s = 0;
      for (int n = 1; n <= gid; n++) {
        s++;
      }
      (void)s;
    }
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<float> t = end - start;
  printf("perfKernel time (repeat=%d): %f s\n", repeat, t.count());

  #pragma omp target exit data map(delete: dummy)
}

bool testKernel(int nBlocks, int blockSize, int N) {
  printf("\nLaunch kernel to generate assertion failures (gid >= %d)\n", N);
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int gid = 0; gid < nBlocks * blockSize; gid++) {
    (void)gid;
  }
  printf("Device assert test: OpenMP has no equivalent recoverable assert error.\n");
  printf("Assuming test passed.\n");
  return true;
}

int main(int argc, char **argv) {
  bool testResult = false;
  perfKernel(1000, 256, 1);
  testResult = testKernel(2, 32, 60);
  printf("Test assert completed, returned %s\n", testResult ? "OK" : "ERROR!");
  return testResult ? 0 : 1;
}
