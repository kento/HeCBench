// OpenMP target offloading port of streamPriority benchmark
// Simulates stream priority by ordering heavy vs light kernels

#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

#define EACH_SIZE (256 * 1024)

static void mem_init(int *buf, size_t n) {
  for (size_t i = 0; i < n; i++)
    buf[i] = (int)i;
}

static void heavy_copy(int *dst, int *src, size_t n) {
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (size_t i = 0; i < n; i++) {
    int v = src[i];
    for (int k = 0; k < 4; k++) {
      if (v > 0) v--;
    }
    dst[i] = src[i];
  }
}

static void light_copy(int *dst, int *src, size_t n) {
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (size_t i = 0; i < n; i++) {
    dst[i] = src[i];
  }
}

static long eval(bool use_priority) {
  const size_t size  = 1UL << 29; // 512 MiB of ints
  const size_t nelem = size / sizeof(int);

  std::vector<int> h_src_low(nelem), h_src_hi(nelem);
  mem_init(h_src_low.data(), nelem);
  mem_init(h_src_hi.data(), nelem);

  int *d_src_low = (int *)malloc(nelem * sizeof(int));
  int *d_src_hi  = (int *)malloc(nelem * sizeof(int));
  int *d_dst_low = (int *)malloc(nelem * sizeof(int));
  int *d_dst_hi  = (int *)malloc(nelem * sizeof(int));

  memcpy(d_src_low, h_src_low.data(), nelem * sizeof(int));
  memcpy(d_src_hi,  h_src_hi.data(),  nelem * sizeof(int));
  memset(d_dst_low, 0, nelem * sizeof(int));
  memset(d_dst_hi,  0, nelem * sizeof(int));

  #pragma omp target enter data map(to: d_src_low[0:nelem], d_src_hi[0:nelem]) \
                                  map(alloc: d_dst_low[0:nelem], d_dst_hi[0:nelem])

  const size_t chunk   = EACH_SIZE / sizeof(int);
  const size_t nchunks = nelem / chunk;

  // Warmup
  for (size_t c = 0; c < nchunks; c++) {
    heavy_copy(d_dst_low + c*chunk, d_src_low + c*chunk, chunk);
    light_copy(d_dst_hi  + c*chunk, d_src_hi  + c*chunk, chunk);
  }

  auto start = std::chrono::steady_clock::now();
  for (size_t c = 0; c < nchunks; c++) {
    if (use_priority) {
      light_copy(d_dst_hi  + c*chunk, d_src_hi  + c*chunk, chunk);
      heavy_copy(d_dst_low + c*chunk, d_src_low + c*chunk, chunk);
    } else {
      heavy_copy(d_dst_low + c*chunk, d_src_low + c*chunk, chunk);
      light_copy(d_dst_hi  + c*chunk, d_src_hi  + c*chunk, chunk);
    }
  }
  auto end = std::chrono::steady_clock::now();

  #pragma omp target update from(d_dst_low[0:nelem], d_dst_hi[0:nelem])
  #pragma omp target exit data map(delete: d_src_low[0:nelem], d_src_hi[0:nelem], \
                                           d_dst_low[0:nelem], d_dst_hi[0:nelem])

  bool ok = true;
  for (size_t i = 0; i < nelem && ok; i++) {
    if (d_dst_low[i] != h_src_low[i] || d_dst_hi[i] != h_src_hi[i]) ok = false;
  }
  if (!ok) fprintf(stderr, "Verification failed!\n");

  free(d_src_low); free(d_src_hi);
  free(d_dst_low); free(d_dst_hi);

  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

int main(int argc, char **argv) {
  printf("Starting [%s]...\n", argv[0]);
  printf("Stream priority range: low: 0 to high: -1 (simulated)\n");

  auto t1 = eval(true);
  printf("Elapsed time of kernel launched to high priority stream: %.3lf ms\n", t1 * 1e-6);

  auto t2 = eval(false);
  printf("Elapsed time of kernel launched to no-priority stream: %.3lf ms\n", t2 * 1e-6);
  return 0;
}
