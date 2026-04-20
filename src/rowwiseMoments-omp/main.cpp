// OpenMP target offloading port of rowwiseMoments benchmark.
// Per-row mean and reciprocal std-dev using Welford's online algorithm.

#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>

int main(int argc, char* argv[])
{
  if (argc != 7) {
    printf("Usage: %s <batch> <channel> <width> <height> <group> <repeat>\n", argv[0]);
    return 1;
  }
  const int N      = atoi(argv[1]);
  const int C      = atoi(argv[2]);
  const int W      = atoi(argv[3]);
  const int H      = atoi(argv[4]);
  const int G      = atoi(argv[5]);
  const int repeat = atoi(argv[6]);

  const long D   = C / G;
  const float eps = 1e-6f;

  const long input_size  = (long)N * C * W * H;
  const long output_size = (long)N * G;

  float* h_X    = (float*)malloc(input_size  * sizeof(float));
  float* h_mean = (float*)malloc(output_size * sizeof(float));
  float* h_rstd = (float*)malloc(output_size * sizeof(float));

  srand(123);
  for (long i = 0; i < input_size; i++)
    h_X[i] = (float)rand() / RAND_MAX;

  float* d_X    = new float[input_size];
  float* d_mean = new float[output_size];
  float* d_rstd = new float[output_size];

  for (long i = 0; i < input_size; i++) d_X[i] = h_X[i];

#pragma omp target enter data map(to: d_X[0:input_size]) \
                              map(alloc: d_mean[0:output_size], d_rstd[0:output_size])

  const long row_len  = D * H * W;
  const int  num_rows = N * G;

  auto t_start = std::chrono::steady_clock::now();

  for (int iter = 0; iter < repeat; iter++) {
#pragma omp target teams distribute parallel for thread_limit(256)
    for (int row = 0; row < num_rows; row++) {
      float mean = 0.f, m2 = 0.f, nf = 0.f;
      for (long j = 0; j < row_len; j++) {
        float x     = d_X[row * row_len + j];
        nf          += 1.f;
        float delta  = x - mean;
        mean         += delta / nf;
        float delta2 = x - mean;
        m2           += delta * delta2;
      }
      float divisor = nf > 0.f ? nf : 1.f;
      d_rstd[row]   = 1.f / sqrtf(m2 / divisor + eps);
      d_mean[row]   = mean;
    }
  }

  auto t_end = std::chrono::steady_clock::now();
  auto time  = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
  printf("Average execution time of RowwiseMoments kernel: %f (us)\n",
         (float)time * 1e-3f / repeat);

#pragma omp target update from(d_mean[0:output_size], d_rstd[0:output_size])
  for (long i = 0; i < output_size; i++) {
    h_mean[i] = d_mean[i];
    h_rstd[i] = d_rstd[i];
  }

  double avg_mean = 0.0, avg_rstd = 0.0;
  for (long i = 0; i < output_size; i++) {
    avg_mean += h_mean[i];
    avg_rstd += h_rstd[i];
  }
  avg_mean /= output_size;
  avg_rstd /= output_size;
  printf("Checksum: mean = %lf and rstd = %lf\n", avg_mean, avg_rstd);

#pragma omp target exit data map(delete: d_X[0:input_size], d_mean[0:output_size], d_rstd[0:output_size])
  free(h_X); free(h_mean); free(h_rstd);
  delete[] d_X; delete[] d_mean; delete[] d_rstd;
  return 0;
}
