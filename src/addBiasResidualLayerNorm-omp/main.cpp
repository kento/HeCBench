#include <algorithm>
#include <chrono>
#include <random>
#include <cstdio>
#include <cmath>
#include <omp.h>

void addBiasResidualLayerNorm(
    float* out, float* input, float* bias, float* gamma, float* beta,
    float eps, int m, int n, int repeat)
{
  auto start = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    #pragma omp target teams distribute num_teams(m) thread_limit(256)
    for (int row = 0; row < m; row++) {
      float mean_sum = 0.f;
      #pragma omp parallel for reduction(+:mean_sum)
      for (int col = 0; col < n; col++) {
        float v = out[row * n + col] + input[row * n + col] + bias[col];
        out[row * n + col] = v;
        mean_sum += v;
      }
      float mean = mean_sum / n;

      float var_sum = 0.f;
      #pragma omp parallel for reduction(+:var_sum)
      for (int col = 0; col < n; col++) {
        float d = out[row * n + col] - mean;
        var_sum += d * d;
      }
      float inv_std = 1.f / sqrtf(var_sum / n + eps);

      #pragma omp parallel for
      for (int col = 0; col < n; col++) {
        float v = (out[row * n + col] - mean) * inv_std;
        out[row * n + col] = v * gamma[col] + beta[col];
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of AddBiasResidualLayerNorm (m=%d, n=%d): %f (us)\n",
         m, n, (time * 1e-3f) / repeat);
}

void layer(int repeat) {
  const int m = 4096;
  int dims[] = {256, 512, 768, 1024, 2048, 4096, 8192};

  for (int dim : dims) {
    const int n = dim;
    std::mt19937 gen(19937);
    std::uniform_real_distribution<float> dis(0.f, 1.f);

    float* h_output = (float*)malloc(m * n * sizeof(float));
    float* h_input  = (float*)malloc(m * n * sizeof(float));
    float* h_bias   = (float*)malloc(n * sizeof(float));
    float* h_gamma  = (float*)malloc(n * sizeof(float));
    float* h_beta   = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < m * n; i++) { h_input[i] = dis(gen); h_output[i] = 0.f; }
    for (int i = 0; i < n; i++) {
      h_bias[i]  = dis(gen);
      h_gamma[i] = dis(gen);
      h_beta[i]  = dis(gen);
    }

    #pragma omp target enter data \
      map(to: h_output[0:m*n], h_input[0:m*n], h_bias[0:n], h_gamma[0:n], h_beta[0:n])

    float eps = 1e-6f;
    addBiasResidualLayerNorm(h_output, h_input, h_bias, h_gamma, h_beta, eps, m, n, repeat);

    #pragma omp target exit data map(from: h_output[0:m*n]) \
      map(delete: h_input[0:m*n], h_bias[0:n], h_gamma[0:n], h_beta[0:n])

    float s = 0;
    for (int i = 0; i < m * n; i++) s += h_output[i];
    printf("Checksum = %f\n", s / ((float)n * n));

    free(h_output); free(h_input); free(h_bias); free(h_gamma); free(h_beta);
  }
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  int repeat = atoi(argv[1]);
  printf("---------------- float32 (OMP target) -------------\n");
  layer(repeat);
  return 0;
}
