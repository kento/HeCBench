// OpenMP target offloading port of wedford (Welford) benchmark
// Per-feature mean and biased variance via Welford's online algorithm

#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("Usage: %s <batch_size> <spatial_size> <feature_size> <repeat>\n", argv[0]);
    return 1;
  }
  const int batch_size   = atoi(argv[1]);
  const int spatial_size = atoi(argv[2]);
  const int feature_size = atoi(argv[3]);
  const int repeat       = atoi(argv[4]);

  const size_t input_size = (size_t)batch_size * spatial_size * feature_size;
  std::vector<float> h_input(input_size);
  srand(123);
  for (size_t i = 0; i < input_size; i++) h_input[i] = rand() / (float)RAND_MAX;

  std::vector<float> h_mean(feature_size, 0.f), h_var(feature_size, 0.f);

  float *d_input = h_input.data();
  float *d_mean  = h_mean.data();
  float *d_var   = h_var.data();

  #pragma omp target enter data map(to: d_input[0:input_size]) \
                                  map(alloc: d_mean[0:feature_size], d_var[0:feature_size])

  auto start = std::chrono::steady_clock::now();

  for (int rep = 0; rep < repeat; rep++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int f = 0; f < feature_size; f++) {
      int   count  = 0;
      float x_mean = 0.f;
      float m_2_n  = 0.f;

      for (int b = 0; b < batch_size; b++) {
        int base = f * spatial_size + b * spatial_size * feature_size;
        for (int s = 0; s < spatial_size; s++) {
          count++;
          float x_n = d_input[base + s];
          float d   = x_n - x_mean;
          x_mean   += d / count;
          m_2_n    += d * (x_n - x_mean);
        }
      }
      d_mean[f] = x_mean;
      d_var [f] = (count > 0) ? m_2_n / count : 0.f;
    }
  }

  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (ms)\n", time * 1e-6f / repeat);

  #pragma omp target update from(d_mean[0:feature_size], d_var[0:feature_size])
  #pragma omp target exit data map(delete: d_input[0:input_size], \
                                           d_mean[0:feature_size], d_var[0:feature_size])

  double avg_var = 0.0, avg_mean = 0.0;
  for (int i = 0; i < feature_size; i++) {
    avg_var  += h_var[i];
    avg_mean += h_mean[i];
  }
  avg_var  /= feature_size;
  avg_mean /= feature_size;
  printf("Checksum: mean = %f and variance = %f\n", avg_var, avg_mean);
  return 0;
}
