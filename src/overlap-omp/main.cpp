#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

static constexpr int N            = 1 << 22;
static constexpr int nreps        = 10;
static constexpr int inner_reps   = 5;
static constexpr int STREAM_COUNT = 4;

// Runs the increment kernel for `streams_used` independent arrays, nreps times.
// Each element: out[idx] = in[idx] + inner_reps  (simulating inner_reps increments).
float processWithOMP(int streams_used,
                     int *d_in[STREAM_COUNT],
                     int *d_out[STREAM_COUNT])
{
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < nreps; i++) {
    for (int s = 0; s < streams_used; s++) {
      int *in_s  = d_in[s];
      int *out_s = d_out[s];
#pragma omp target teams distribute parallel for thread_limit(256) \
    map(always,to: in_s[0:N]) map(always,from: out_s[0:N])
      for (int idx = 0; idx < N; idx++) {
        int val = in_s[idx];
        for (int k = 0; k < inner_reps; ++k) val += 1;
        out_s[idx] = val;
      }
    }
  }
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-6f;
}

int main(int, char *[])
{
  printf("Length of the array = %d\n", N);

  int *d_in[STREAM_COUNT];
  int *d_out[STREAM_COUNT];
  for (int s = 0; s < STREAM_COUNT; s++) {
    d_in[s]  = (int *)malloc(N * sizeof(int));
    d_out[s] = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) { d_in[s][i] = 0; d_out[s][i] = 0; }
#pragma omp target enter data map(to: d_in[s][0:N], d_out[s][0:N])
  }

  float serial_time  = processWithOMP(1,            d_in, d_out);
  float overlap_time = processWithOMP(STREAM_COUNT, d_in, d_out);

  printf("\nAverage measured timings over %d repetitions:\n", nreps);
  printf(" Avg. time when execution fully serialized\t: %f ms\n", serial_time / nreps);
  printf(" Avg. time when overlapped using %d streams\t: %f ms\n", STREAM_COUNT, overlap_time / nreps);
  printf(" Avg. speedup gained (serialized - overlapped)\t: %f\n", (serial_time - overlap_time) / nreps);
  printf("\nMeasured throughput:\n");
  int memsize = N * (int)sizeof(int);
  printf(" Fully serialized execution\t\t: %f GB/s\n",
         (nreps * (memsize * 2e-6)) / serial_time);
  printf(" Overlapped using %d streams\t\t: %f GB/s\n",
         STREAM_COUNT, (nreps * (memsize * 2e-6)) / overlap_time);

  bool passed = true;
  for (int s = 0; s < STREAM_COUNT && passed; s++) {
#pragma omp target update from(d_out[s][0:N])
    for (int i = 0; i < N && passed; i++)
      passed = (d_out[s][i] == inner_reps);
  }
  printf("\n%s\n", passed ? "PASS" : "FAIL");

  for (int s = 0; s < STREAM_COUNT; s++) {
#pragma omp target exit data map(delete: d_in[s][0:N], d_out[s][0:N])
    free(d_in[s]);
    free(d_out[s]);
  }
  return 0;
}
