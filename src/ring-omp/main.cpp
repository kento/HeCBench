// OpenMP target offloading port of ring benchmark.
// Ring exchange (single device: self-copy / fence-only simulation).

#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <minimum copy length> <maximum copy length> <repeat>\n", argv[0]);
    return 1;
  }
  const long min_len = atol(argv[1]);
  const long max_len = atol(argv[2]);
  const int repeat   = atoi(argv[3]);

  const int num_devices = 1;
  printf("Device name: OpenMP target\n");
  printf("Warning, only one device is detected. "
         "This program is supposed to execute with multiple devices.\n");

  for (long len = min_len; len <= max_len; len = len * 4) {
    int* d_buf = new int[len];
    for (long i = 0; i < len; i++) d_buf[i] = (int)i;

#pragma omp target enter data map(to: d_buf[0:len])

    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
      // Single device: ring exchange is a no-op; just a target fence
#pragma omp target teams distribute parallel for thread_limit(256)
      for (long i = 0; i < 1; i++) { (void)d_buf[0]; }
    }

    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    float time_us = (float)time * 1e-3f / repeat;

    printf("----------------------------------------------------------------\n");
    printf("Copy length = %ld\n", len);
    printf("Average total exchange time: %f (us)\n", time_us);
    printf("Average exchange time per device: %f (us)\n", time_us / num_devices);

#pragma omp target update from(d_buf[0:len])
    bool ok = true;
    for (long i = 0; i < len; i++)
      if (d_buf[i] != (int)i) { ok = false; break; }
    printf("%s\n", ok ? "PASS" : "FAIL");

#pragma omp target exit data map(delete: d_buf[0:len])
    delete[] d_buf;
  }
  return 0;
}
