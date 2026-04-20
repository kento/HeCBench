/*
 * OpenMP target offloading port of blas-gemmEx2 (cublasLt-based GEMM).
 */

#include <sys/time.h>
#include <stdio.h>
#include <cstdint>
#include <chrono>
#include <vector>
#include <omp.h>

static int8_t float2int8(float f, float scale) {
  int8_t i = (int8_t)(f * scale);
  if (i < -127) i = -127;
  if (i > 127)  i = 127;
  return i;
}

static void performance(int m, int n, int k, bool is_integer, double avg_time_ms) {
  double total_ops = (double)m * n * k * 2;
  double perf = (total_ops / avg_time_ms) * 1e-9 * 1e3;
  const char* scale = "G";
  const char* unit = is_integer ? "OP/s" : "FLOP/s";
  if (perf >= 1000) { perf /= 1000; scale = "T"; }
  printf("%lf %s%s\n", perf, scale, unit);
}

template<typename T, typename S>
void gemm_omp(T* A, T* B, S* C, int m, int n, int k) {
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int idx = 0; idx < m * n; idx++) {
    int col = idx / m;
    int row = idx % m;
    S s = S(0);
    for (int i = 0; i < k; i++)
      s += (S)A[row + i * m] * (S)B[i + col * k];
    C[row + col * m] = s;
  }
}

template<typename T, typename S>
void test_gemm(T* A, T* B, S* C, int m, int n, int k,
               int iteration, bool is_int = false, int compute_mode = 0) {
  double total_time = 0;
  for (int i = 0; i < iteration; i++) {
    auto start = std::chrono::steady_clock::now();
    gemm_omp(A, B, C, m, n, k);
    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         end - start).count() * 1e-6;
    if (i > 0) total_time += elapsed;
  }
  double avg_time = total_time / (iteration - 1);
  printf("mode %d: %.3f ms\n", compute_mode, avg_time);
  performance(m, n, k, is_int, avg_time);
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <M> <N> <K> <iterations>\n", argv[0]);
    return 1;
  }
  const int m = atoi(argv[1]), n = atoi(argv[2]), k = atoi(argv[3]);
  const int iteration = atoi(argv[4]);
  printf("shape: (%d, %d) x (%d, %d)\n", m, k, k, n);

  std::vector<double>  h_dA(m*k), h_dB(k*n);
  std::vector<float>   h_fA(m*k), h_fB(k*n);
  std::vector<int8_t>  h_iA(m*k), h_iB(k*n);

  for (int i = 0; i < m*k; i++) {
    float v = (float)(i % 255 - 127) / 127.f;
    h_dA[i] = (double)v; h_fA[i] = v; h_iA[i] = float2int8(v, 127.f);
  }
  for (int i = 0; i < k*n; i++) {
    float v = (float)(i % 255 - 127) / 127.f;
    h_dB[i] = (double)v; h_fB[i] = v; h_iB[i] = float2int8(v, 127.f);
  }

  double*  dA = (double*)malloc(m*k*sizeof(double));
  double*  dB = (double*)malloc(k*n*sizeof(double));
  double*  dC = (double*)calloc(m*n, sizeof(double));
  float*   fA = (float*)malloc(m*k*sizeof(float));
  float*   fB = (float*)malloc(k*n*sizeof(float));
  float*   fC = (float*)calloc(m*n, sizeof(float));
  int32_t* iC = (int32_t*)calloc(m*n, sizeof(int32_t));
  int8_t*  iA = (int8_t*)malloc(m*k*sizeof(int8_t));
  int8_t*  iB = (int8_t*)malloc(k*n*sizeof(int8_t));

  for (int i = 0; i < m*k; i++) { dA[i]=h_dA[i]; fA[i]=h_fA[i]; iA[i]=h_iA[i]; }
  for (int i = 0; i < k*n; i++) { dB[i]=h_dB[i]; fB[i]=h_fB[i]; iB[i]=h_iB[i]; }

  #pragma omp target enter data map(to: dA[0:m*k], dB[0:k*n], fA[0:m*k], fB[0:k*n], \
                                        iA[0:m*k], iB[0:k*n]) \
                                map(alloc: dC[0:m*n], fC[0:m*n], iC[0:m*n])

  printf(">>>>>>>>>>>>>>>>> test fp64 >>>>>>>>>>>>>>>>>\n");
  test_gemm(dA, dB, dC, m, n, k, iteration);

  printf(">>>>>>>>>>>>>>>>> test fp32 (compute type tf32) >>>>>>>>>>>>>>>>>\n");
  test_gemm(fA, fB, fC, m, n, k, iteration, false, 3);

  printf(">>>>>>>>>>>>>>>>> test fp32 (compute type bf16) >>>>>>>>>>>>>>>>>\n");
  test_gemm(fA, fB, fC, m, n, k, iteration, false, 2);

  printf(">>>>>>>>>>>>>>>>> test fp32 (compute type fp16) >>>>>>>>>>>>>>>>>\n");
  test_gemm(fA, fB, fC, m, n, k, iteration, false, 1);

  printf(">>>>>>>>>>>>>>>>> test fp32 (compute type fp32) >>>>>>>>>>>>>>>>>\n");
  test_gemm(fA, fB, fC, m, n, k, iteration, false, 0);

  printf(">>>>>>>>>>>>>>>>> test int8 >>>>>>>>>>>>>>>>>\n");
  test_gemm(iA, iB, iC, m, n, k, iteration, true);

  printf(">>>>>>>>>>>>>>>>> compare first ten values >>>>>>>>>>>>>>>>>\n");
  #pragma omp target update from(dC[0:m*n], fC[0:m*n], iC[0:m*n])
  printf("fp64: "); for (int i = 0; i < 10; i++) printf("%.5lf%c", dC[i], " \n"[i==9]);
  printf("fp32: "); for (int i = 0; i < 10; i++) printf("%.5f%c",  fC[i], " \n"[i==9]);
  printf("int8: "); for (int i = 0; i < 10; i++) printf("%.5f%c",  (float)iC[i]/127/127, " \n"[i==9]);

  #pragma omp target exit data map(delete: dA[0:m*k], dB[0:k*n], fA[0:m*k], fB[0:k*n], \
                                           iA[0:m*k], iB[0:k*n], dC[0:m*n], fC[0:m*n], iC[0:m*n])
  free(dA); free(dB); free(dC); free(fA); free(fB); free(fC); free(iC); free(iA); free(iB);
  return 0;
}
