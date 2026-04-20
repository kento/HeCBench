// OpenMP target offloading port of zerocopy benchmark
// Vector add using target data map (equivalent of zero-copy memory)

#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

static void eval(bool warmup, const int repeat) {
  if (warmup) printf("Warmup...\n");

  for (int nelem = 1024*1024; nelem <= 1024*1024*64; nelem *= 2) {
    if (!warmup) printf("\nvector length = %d\n", nelem);

    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> a(nelem), b(nelem), c(nelem, 0.f);
    auto t1 = std::chrono::steady_clock::now();
    if (!warmup)
      printf("Memory allocation (host vector): %lf ms\n",
             std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count() * 1e-6);

    srand(nelem);
    for (int i = 0; i < nelem; i++) {
      a[i] = rand() / (float)RAND_MAX;
      b[i] = rand() / (float)RAND_MAX;
    }

    float *h_a = a.data(), *h_b = b.data(), *h_c = c.data();

    if (!warmup) {
      t0 = std::chrono::steady_clock::now();
      t1 = std::chrono::steady_clock::now();
      printf("cudaHostGetDevicePointer: %lf ms\n",
             std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count() * 1e-6);
    }

    // Map host memory directly to device
    t0 = std::chrono::steady_clock::now();
    #pragma omp target data map(to: h_a[0:nelem], h_b[0:nelem]) map(tofrom: h_c[0:nelem])
    {
      for (int r = 0; r < repeat; r++) {
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int i = 0; i < nelem; i++)
          h_c[i] = h_a[i] + h_b[i];
      }
    }
    t1 = std::chrono::steady_clock::now();
    if (!warmup)
      printf("Average kernel execution time: %lf ms\n",
             std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count() * 1e-6 / repeat);

    if (warmup) {
      float errorNorm = 0.f, refNorm = 0.f;
      for (int i = 0; i < nelem; i++) {
        float ref  = a[i] + b[i];
        float diff = c[i] - ref;
        errorNorm += diff * diff;
        refNorm   += ref  * ref;
      }
      errorNorm = sqrtf(errorNorm);
      refNorm   = sqrtf(refNorm);
      printf("%s\n", (errorNorm / refNorm < 1e-6f) ? "SUCCESS" : "FAILURE");
    }

    if (!warmup) {
      t0 = std::chrono::steady_clock::now();
      a.clear(); a.shrink_to_fit();
      b.clear(); b.shrink_to_fit();
      c.clear(); c.shrink_to_fit();
      t1 = std::chrono::steady_clock::now();
      printf("Memory deallocation (host vector): %lf ms\n",
             std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count() * 1e-6);
    }
  }
  if (warmup) printf("Done.\n");
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  printf("> Using Host Allocated (cudaHostAlloc)\n");
  eval(true,  repeat);
  eval(false, repeat);

  printf("> Using Generic System Paged Memory (malloc)\n");
  eval(true,  repeat);
  eval(false, repeat);
  return 0;
}
