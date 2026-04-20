// OpenMP target offloading port of qkv benchmark.
// Matrix multiply: out(BT, OC) = inp(BT, C) * weight^T(OC, C) + bias(OC)
// Two kernel variants: naive (parallel over BT*OC) and tiled (team-based).

#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

static float* make_random_float(size_t n) {
  float* arr = new float[n];
  for (size_t i = 0; i < n; i++)
    arr[i] = (float)rand() / (float)RAND_MAX * 2.f - 1.f;
  return arr;
}

static void matmul_cpu(float* out,
                       const float* inp, const float* weight, const float* bias,
                       int BT, int C, int OC) {
  for (int bt = 0; bt < BT; bt++)
    for (int oc = 0; oc < OC; oc++) {
      float val = bias ? bias[oc] : 0.f;
      for (int c = 0; c < C; c++)
        val += inp[bt * C + c] * weight[oc * C + c];
      out[bt * OC + oc] = val;
    }
}

// Kernel 1: naive parallel over BT*OC
static void matmul_kernel1(float* d_out, const float* d_inp,
                            const float* d_weight, const float* d_bias,
                            int BT, int C, int OC) {
#pragma omp target teams distribute parallel for thread_limit(256)
  for (int bt = 0; bt < BT * OC; bt++) {
    int b = bt / OC, oc = bt % OC;
    float val = d_bias[oc];
    for (int c = 0; c < C; c++)
      val += d_inp[b * C + c] * d_weight[oc * C + c];
    d_out[b * OC + oc] = val;
  }
}

// Kernel 4: tiled – each team handles one BT row
static void matmul_kernel4(float* d_out, const float* d_inp,
                            const float* d_weight, const float* d_bias,
                            int BT, int C, int OC) {
#pragma omp target teams distribute num_teams(BT) thread_limit(256)
  for (int bt = 0; bt < BT; bt++) {
#pragma omp parallel for
    for (int oc = 0; oc < OC; oc++) {
      float val = d_bias[oc];
      for (int c = 0; c < C; c++)
        val += d_inp[bt * C + c] * d_weight[oc * C + c];
      d_out[bt * OC + oc] = val;
    }
  }
}

static void add_bias_kernel(float* d_out, const float* d_bias, int BT, int OC) {
#pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < BT * OC; i++)
    d_out[i] += d_bias[i % OC];
}

int main(int argc, char** argv) {
  if (argc != 5) {
    printf("Usage: %s <B> <T> <C> <repeat>\n", argv[0]);
    return 1;
  }
  const int B      = atoi(argv[1]);
  const int T      = atoi(argv[2]);
  const int C      = atoi(argv[3]);
  const int repeat = atoi(argv[4]);
  const int OC     = C * 4;
  const int BT     = B * T;

  printf("B=%d T=%d C=%d OC=%d repeat=%d\n", B, T, C, OC, repeat);

  srand(0);
  float* h_inp    = make_random_float((size_t)BT * C);
  float* h_weight = make_random_float((size_t)OC * C);
  float* h_bias   = make_random_float(OC);
  float* h_ref    = new float[(size_t)BT * OC]();

  matmul_cpu(h_ref, h_inp, h_weight, h_bias, BT, C, OC);

  size_t inp_sz    = (size_t)BT * C;
  size_t weight_sz = (size_t)OC * C;
  size_t out_sz    = (size_t)BT * OC;

  float* d_inp    = new float[inp_sz];
  float* d_weight = new float[weight_sz];
  float* d_bias   = new float[OC];
  float* d_out    = new float[out_sz];

  memcpy(d_inp,    h_inp,    inp_sz    * sizeof(float));
  memcpy(d_weight, h_weight, weight_sz * sizeof(float));
  memcpy(d_bias,   h_bias,   OC        * sizeof(float));

#pragma omp target enter data map(to: d_inp[0:inp_sz], d_weight[0:weight_sz], d_bias[0:OC]) \
                              map(alloc: d_out[0:out_sz])

  // ----- Kernel 1 -----
  printf("\n--- Kernel 1 (naive) ---\n");
  matmul_kernel1(d_out, d_inp, d_weight, d_bias, BT, C, OC);
#pragma omp target update from(d_out[0:out_sz])
  {
    int errs = 0;
    for (size_t i = 0; i < out_sz; i++)
      if (std::fabs(d_out[i] - h_ref[i]) > 1e-1f && ++errs <= 5)
        printf("  Mismatch k1[%zu]: got %.6f ref %.6f\n", i, d_out[i], h_ref[i]);
    printf(errs ? "  FAIL: %d mismatches\n" : "  All results match for kernel1.\n", errs);
  }

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++)
    matmul_kernel1(d_out, d_inp, d_weight, d_bias, BT, C, OC);
#pragma omp taskwait
  auto t1 = std::chrono::steady_clock::now();
  double ms1 = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-6 / repeat;
  double tflops1 = (double)BT * OC * C * 2 / ms1 * 1e-9;
  printf("  time %.4f ms | tflops %.2f\n", ms1, tflops1);

  // ----- Kernel 4 -----
  printf("\n--- Kernel 4 (tiled) ---\n");
  matmul_kernel4(d_out, d_inp, d_weight, d_bias, BT, C, OC);
#pragma omp target update from(d_out[0:out_sz])
  {
    int errs = 0;
    for (size_t i = 0; i < out_sz; i++)
      if (std::fabs(d_out[i] - h_ref[i]) > 1e-1f && ++errs <= 5)
        printf("  Mismatch k4[%zu]: got %.6f ref %.6f\n", i, d_out[i], h_ref[i]);
    printf(errs ? "  FAIL: %d mismatches\n" : "  All results match for kernel4.\n", errs);
  }

  t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++)
    matmul_kernel4(d_out, d_inp, d_weight, d_bias, BT, C, OC);
  t1 = std::chrono::steady_clock::now();
  double ms4 = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-6 / repeat;
  double tflops4 = (double)BT * OC * C * 2 / ms4 * 1e-9;
  printf("  time %.4f ms | tflops %.2f\n", ms4, tflops4);

  // ----- add_bias -----
  printf("\n--- add_bias (standalone) ---\n");
  t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++)
    add_bias_kernel(d_out, d_bias, BT, OC);
  t1 = std::chrono::steady_clock::now();
  double ms_ab = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-6 / repeat;
  printf("  time %.4f ms\n", ms_ab);

#pragma omp target exit data map(delete: d_inp[0:inp_sz], d_weight[0:weight_sz], \
                                         d_bias[0:OC], d_out[0:out_sz])
  delete[] h_inp; delete[] h_weight; delete[] h_bias; delete[] h_ref;
  delete[] d_inp; delete[] d_weight; delete[] d_bias; delete[] d_out;
  return 0;
}
