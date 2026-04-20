#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv)
{
  printf("[%s] - Starting...\n", argv[0]);
  if (argc < 2) { printf("Usage: %s <repeat>\n", argv[0]); return 1; }
  const int repeat = atoi(argv[1]);

  const int buf_len  = 1024 * 1024 * 16;
  const size_t buf_size = (size_t)buf_len * sizeof(float);

  float *d_buf = (float *)malloc(buf_size);
  float *d_tmp = (float *)malloc(buf_size);
  float *h_buf = (float *)malloc(buf_size);

  for (int i = 0; i < buf_len; i++) { d_buf[i] = 0.0f; d_tmp[i] = 0.0f; }
#pragma omp target enter data map(alloc: d_buf[0:buf_len], d_tmp[0:buf_len])
#pragma omp target update to(d_buf[0:buf_len], d_tmp[0:buf_len])

  // Measure device-to-device copy bandwidth using parallel kernels.
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    if (i % 2 == 0) {
#pragma omp target teams distribute parallel for thread_limit(256)
      for (int j = 0; j < buf_len; j++) d_tmp[j] = d_buf[j];
    } else {
#pragma omp target teams distribute parallel for thread_limit(256)
      for (int j = 0; j < buf_len; j++) d_buf[j] = d_tmp[j];
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  printf("Single-device copy bandwidth: %.2f GB/s\n",
         1.0 / (double)time_ns * (double)((long long)repeat * (long long)buf_size));

  // Initialize host buffer, upload to device, run two kernel passes, verify.
  printf("Preparing host buffer and copying to device...\n");
  for (int i = 0; i < buf_len; ++i) h_buf[i] = (float)(i % 4096);
  for (int i = 0; i < buf_len; ++i) d_buf[i] = h_buf[i];
#pragma omp target update to(d_buf[0:buf_len])

  printf("Run kernel pass 1: d_tmp[i] = d_buf[i] * 2...\n");
#pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < buf_len; i++) d_tmp[i] = d_buf[i] * 2.0f;

  printf("Run kernel pass 2: d_buf[i] = d_tmp[i] * 2...\n");
#pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < buf_len; i++) d_buf[i] = d_tmp[i] * 2.0f;

  printf("Copying result back to host and verifying...\n");
#pragma omp target update from(d_buf[0:buf_len])

  int error_count = 0;
  for (int i = 0; i < buf_len; ++i) {
    float ref = (float)(i % 4096) * 4.0f;
    if (d_buf[i] != ref) {
      printf("Verification error @ element %d: val = %f, ref = %f\n",
             i, (double)d_buf[i], (double)ref);
      if (++error_count > 10) break;
    }
  }
  if (error_count == 0) printf("Test passed\n");
  else printf("Test failed!\n");

#pragma omp target exit data map(delete: d_buf[0:buf_len], d_tmp[0:buf_len])
  free(d_buf);
  free(d_tmp);
  free(h_buf);
  return 0;
}
