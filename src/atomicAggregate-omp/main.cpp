/*
 * OpenMP target offloading port of atomicAggregate benchmark.
 */

#include <chrono>
#include <cstdio>
#include <omp.h>

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int nBlocks   = 65536;
  const int blockSize = 256;
  const int N         = nBlocks * blockSize;

  for (int ds = 32; ds >= 1; ds /= 2) {
    int* d = (int*)calloc(ds, sizeof(int));

    #pragma omp target enter data map(to: d[0:ds])

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      #pragma omp target teams distribute parallel for thread_limit(256)
      for (int tid = 0; tid < N; tid++) {
        int idx = tid % ds;
        #pragma omp atomic update
        d[idx]++;
      }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<float> time = end - start;
    printf("Total kernel time (%d locations): %f (s)\n", ds, time.count());

    #pragma omp target update from(d[0:ds])
    #pragma omp target exit data map(delete: d[0:ds])

    int expected = (blockSize / ds) * nBlocks * repeat;
    bool ok = true;
    for (int i = 0; i < ds; i++) {
      if (d[i] != expected) { ok = false; break; }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    free(d);
  }
  return 0;
}
