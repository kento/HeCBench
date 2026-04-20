// OpenMP target offloading port of ert-kokkos (Empirical Roofline Toolkit benchmark)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/time.h>
#include <omp.h>

#define ERT_ALIGN           256
#define ERT_MEMORY_MAX      33554432
#define ERT_WORKING_SET_MIN 128
#define ERT_TRIALS_MIN      1
#define ERT_WSS_MULT        1.3

int gpu_blocks;
int gpu_threads;

double getTime()
{
  struct timeval tm;
  gettimeofday(&tm, NULL);
  return tm.tv_sec + (tm.tv_usec / 1000000.0);
}

template <typename T>
void run(uint64_t PSIZE)
{
  uint64_t nsize = (PSIZE & ~((uint64_t)(ERT_ALIGN - 1))) / sizeof(T);

  T* d_buf = (T*)malloc(nsize * sizeof(T));
  for (uint64_t i = 0; i < nsize; i++) d_buf[i] = (T)(-1);

  #pragma omp target enter data map(tofrom: d_buf[0:nsize])

  uint64_t n = ERT_WORKING_SET_MIN;
  while (n <= nsize) {
    uint64_t ntrials = nsize / n;
    if (ntrials < ERT_TRIALS_MIN) ntrials = ERT_TRIALS_MIN;

    const uint32_t total_threads = (uint32_t)(gpu_blocks * gpu_threads);
    const uint32_t n32 = (uint32_t)n;
    const uint64_t nt  = ntrials;

    #pragma omp target teams distribute parallel for num_teams(gpu_blocks) thread_limit(gpu_threads)
    for (uint32_t tid = 0; tid < total_threads; tid++) {
      if (tid >= n32) continue;
      T alpha = (T)2, const_beta = (T)1;
      for (uint64_t j = 0; j < nt; j++) {
        for (uint32_t i = tid; i < n32; i += total_threads) {
          T beta = const_beta;
          T ai   = d_buf[i];
          beta = ai + alpha;
          beta = beta * ai + alpha;
          beta = beta * ai + alpha;  beta = beta * ai + alpha;
          beta = beta * ai + alpha;  beta = beta * ai + alpha;
          beta = beta * ai + alpha;  beta = beta * ai + alpha;
          beta = beta * ai + alpha;  beta = beta * ai + alpha;
          beta = beta * ai + alpha;  beta = beta * ai + alpha;
          d_buf[i] = -beta;
        }
      }
    }

    uint64_t nNew = (uint64_t)(ERT_WSS_MULT * n);
    if (nNew == n) nNew = n + 1;
    n = nNew;
  }

  #pragma omp target update from(d_buf[0:nsize])

  double checksum = 0;
  for (uint64_t i = 0; i < nsize; i++) checksum += (double)d_buf[i];
  printf("checksum: %lf\n", checksum);

  #pragma omp target exit data map(delete: d_buf[0:nsize])
  free(d_buf);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    fprintf(stderr, "Usage: %s gpu_blocks gpu_threads\n", argv[0]);
    return -1;
  }
  gpu_blocks  = atoi(argv[1]);
  gpu_threads = atoi(argv[2]);
  printf("GPU_BLOCKS  %d\nGPU_THREADS %d\n\n", gpu_blocks, gpu_threads);

  const uint64_t PSIZE = ERT_MEMORY_MAX;
  double start;

  // float (stands in for half2)
  start = getTime();
  run<float>(PSIZE);
  printf("runtime (float/half2 proxy): %lf (s)\n", getTime() - start);

  // float
  start = getTime();
  run<float>(PSIZE);
  printf("runtime (float): %lf (s)\n", getTime() - start);

  // double
  start = getTime();
  run<double>(PSIZE);
  printf("runtime (double): %lf (s)\n", getTime() - start);

  return 0;
}
