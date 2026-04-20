// OpenMP target offloading port of dwconv1d-kokkos (RWKV time-mixing 1D depthwise conv)

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <omp.h>
#include <vector>

void timex_forward(
    const float* w,  // (C*T)
    const float* k,  // (B*C*T)
    float*       x,  // (B*C*T)
    float eps, int B, int C, int T, int repeat)
{
  auto start = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int bc = 0; bc < B * C; bc++) {
      int b = bc / C;
      int c = bc % C;
      for (int t = 0; t < T; t++) {
        float s = eps;
        for (int u = 0; u <= t; u++) {
          s += w[c * T + (T - 1 - (t - u))] * k[b * C * T + c * T + u];
        }
        x[b * C * T + c * T + t] = s;
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("timex forward average time: %f (us)\n", time * 1e-3f / repeat);
}

void timex_backward(
    const float* w,   // (C*T)
    const float* k,   // (B*C*T)
    const float* gwk, // (B*C*T)
    float*       gw,  // (B*C*T)
    float*       gk,  // (B*C*T)
    int B, int C, int T, int repeat)
{
  auto start = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int bc = 0; bc < B * C; bc++) {
      int b = bc / C;
      int c = bc % C;

      for (int t = 0; t < T; t++) {
        float sg = 0.f;
        for (int u = t; u < T; u++) {
          int k_idx = u - (T - 1 - t);
          if (k_idx >= 0 && k_idx < T)
            sg += gwk[b * C * T + c * T + u] * k[b * C * T + c * T + k_idx];
        }
        gw[b * C * T + c * T + t] = sg;
      }

      for (int u = 0; u < T; u++) {
        float sk = 0.f;
        for (int t = u; t < T; t++) {
          sk += gwk[b * C * T + c * T + t] * w[c * T + (T - 1 - (t - u))];
        }
        gk[b * C * T + c * T + u] = sk;
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("timex backward average time: %f (us)\n", time * 1e-3f / repeat);
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <B> <C> <T> <repeat>\n", argv[0]);
    return 1;
  }

  const int B      = atoi(argv[1]);
  const int C      = atoi(argv[2]);
  const int T      = atoi(argv[3]);
  const int repeat = atoi(argv[4]);
  const float eps  = 0.1f;

  const int wSz = C * T;
  const int kSz = B * C * T;

  std::vector<float> h_w(wSz), h_k(kSz), h_gwk(kSz);
  srand(42);
  for (auto& v : h_w)   v = rand() / (float)RAND_MAX;
  for (auto& v : h_k)   v = rand() / (float)RAND_MAX;
  for (auto& v : h_gwk) v = rand() / (float)RAND_MAX;

  float* d_w   = (float*)malloc(wSz * sizeof(float));
  float* d_k   = (float*)malloc(kSz * sizeof(float));
  float* d_x   = (float*)malloc(kSz * sizeof(float));
  float* d_gw  = (float*)malloc(kSz * sizeof(float));
  float* d_gk  = (float*)malloc(kSz * sizeof(float));
  float* d_gwk = (float*)malloc(kSz * sizeof(float));

  for (int i = 0; i < wSz; i++) d_w[i]   = h_w[i];
  for (int i = 0; i < kSz; i++) d_k[i]   = h_k[i];
  for (int i = 0; i < kSz; i++) d_gwk[i] = h_gwk[i];

  #pragma omp target enter data map(to: d_w[0:wSz], d_k[0:kSz], d_gwk[0:kSz]) \
                                map(alloc: d_x[0:kSz], d_gw[0:kSz], d_gk[0:kSz])

  timex_forward(d_w, d_k, d_x, eps, B, C, T, repeat);
  timex_backward(d_w, d_k, d_gwk, d_gw, d_gk, B, C, T, repeat);

  #pragma omp target update from(d_x[0:kSz])

  float sum = 0.f;
  for (int i = 0; i < kSz; i++) sum += d_x[i];
  printf("Forward checksum: %f\n", sum / kSz);

  #pragma omp target exit data map(delete: d_w[0:wSz], d_k[0:kSz], d_gwk[0:kSz], \
                                           d_x[0:kSz], d_gw[0:kSz], d_gk[0:kSz])
  free(d_w); free(d_k); free(d_x); free(d_gw); free(d_gk); free(d_gwk);
  return 0;
}
