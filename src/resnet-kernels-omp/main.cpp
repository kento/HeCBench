// OpenMP target offloading port of resnet-kernels benchmark.
// Winograd convolution (BtdB + GEMM + AtIA) and 1x1 convolution.

#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>

// ============================================================
// RNG helper
// ============================================================
static void fillRandom(float* v, int n, float lo = -0.1f, float hi = 0.1f) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (int i = 0; i < n; i++) v[i] = dist(rng);
}

// ============================================================
// Winograd BtdB: apply B^T * tile * B for each (tile, channel)
// ============================================================
#pragma omp declare target
static float btRow(int r, float d0, float d1, float d2,
                   float d3, float d4, float d5) {
  switch (r) {
    case 0: return  4*d0          - 5*d2          + d4;
    case 1: return        -4*d1   - 4*d2 +  d3   + d4;
    case 2: return         4*d1   - 4*d2 -  d3   + d4;
    case 3: return        -2*d1   -   d2 + 2*d3  + d4;
    case 4: return         2*d1   -   d2 - 2*d3  + d4;
    default:return         4*d1          - 5*d3        + d5;
  }
}

static float atRow(int r, float m0, float m1, float m2,
                   float m3, float m4, float m5) {
  switch (r) {
    case 0: return m0 + m1 + m2 + m3 + m4;
    case 1: return      m1 - m2 + 2*m3 - 2*m4;
    case 2: return      m1 + m2 + 4*m3 + 4*m4;
    default: return     m1 - m2 + 8*m3 - 8*m4 + m5;
  }
}
#pragma omp end declare target

static void winograd_BtdB(const float* input, float* output,
                           int nTilesH, int nTilesW, int C) {
  const int nTiles = nTilesH * nTilesW;
#pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
  for (int tile = 0; tile < nTiles; tile++) {
    for (int c = 0; c < C; c++) {
      float d[6][6];
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
          d[i][j] = input[tile * 36 * C + (i*6 + j) * C + c];

      float tmp[6][6];
      for (int r = 0; r < 6; ++r)
        for (int j = 0; j < 6; ++j)
          tmp[r][j] = btRow(r, d[0][j], d[1][j], d[2][j], d[3][j], d[4][j], d[5][j]);

      float out[6][6];
      for (int i = 0; i < 6; ++i)
        for (int r = 0; r < 6; ++r)
          out[i][r] = btRow(r, tmp[i][0], tmp[i][1], tmp[i][2], tmp[i][3], tmp[i][4], tmp[i][5]);

      for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
          output[(i*6 + j) * nTiles * C + tile * C + c] = out[i][j];
    }
  }
}

static void winograd_gemm(const float* V, const float* U, float* M,
                           int nTiles, int Cin, int Cout) {
#pragma omp target teams distribute parallel for collapse(3) thread_limit(256)
  for (int w = 0; w < 36; w++) {
    for (int tile = 0; tile < nTiles; tile++) {
      for (int cout = 0; cout < Cout; cout++) {
        float sum = 0.f;
        for (int cin = 0; cin < Cin; ++cin)
          sum += V[w * nTiles * Cin + tile * Cin + cin]
               * U[w * Cin * Cout  + cin  * Cout + cout];
        M[w * nTiles * Cout + tile * Cout + cout] = sum;
      }
    }
  }
}

static void winograd_AtIA(const float* M, const float* bias,
                           const float* bnScale, float* output,
                           int nTiles, int Cout) {
#pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
  for (int tile = 0; tile < nTiles; tile++) {
    for (int cout = 0; cout < Cout; cout++) {
      float m[6][6];
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
          m[i][j] = M[(i*6 + j) * nTiles * Cout + tile * Cout + cout];

      float tmp[4][6];
      for (int j = 0; j < 6; ++j) {
        float col[4];
        for (int r = 0; r < 4; ++r)
          col[r] = atRow(r, m[0][j], m[1][j], m[2][j], m[3][j], m[4][j], m[5][j]);
        for (int r = 0; r < 4; ++r) tmp[r][j] = col[r];
      }
      float out[4][4];
      for (int i = 0; i < 4; ++i)
        for (int r = 0; r < 4; ++r)
          out[i][r] = atRow(r, tmp[i][0], tmp[i][1], tmp[i][2], tmp[i][3], tmp[i][4], tmp[i][5]);

      float b = bias[cout], s = bnScale[cout];
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
          float v = s * out[i][j] + b;
          output[tile * 16 * Cout + (i*4 + j) * Cout + cout] = v > 0.f ? v : 0.f;
        }
    }
  }
}

static void conv1x1(const float* A, const float* W,
                    const float* bnBias, const float* bnScale,
                    float* C,
                    int nTiles, int Cin, int Cout) {
#pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
  for (int tile = 0; tile < nTiles; tile++) {
    for (int cout = 0; cout < Cout; cout++) {
      float sum = 0.f;
      for (int cin = 0; cin < Cin; ++cin)
        sum += A[tile * Cin + cin] * W[cin * Cout + cout];
      float v = bnScale[cout] * sum + bnBias[cout];
      C[tile * Cout + cout] = v > 0.f ? v : 0.f;
    }
  }
}

// ============================================================
struct KernelResult { double time_us; double ktime_us; };

static KernelResult kernel_winograd(int repeat, int Cin, int Cout) {
  const int H = 14, W = 14;
  const int nTH = (H - 2) / 2;
  const int nTW = (W - 2) / 2;
  const int nTiles = nTH * nTW;

  int input_sz  = nTiles * 36 * Cin;
  int V_sz      = 36 * nTiles * Cin;
  int U_sz      = 36 * Cin * Cout;
  int M_sz      = 36 * nTiles * Cout;
  int out_sz    = nTiles * 16 * Cout;

  float* d_input  = new float[input_sz];
  float* d_V      = new float[V_sz];
  float* d_U      = new float[U_sz];
  float* d_M      = new float[M_sz];
  float* d_bias   = new float[Cout];
  float* d_scale  = new float[Cout];
  float* d_output = new float[out_sz];

  fillRandom(d_input, input_sz);
  fillRandom(d_V, V_sz);
  fillRandom(d_U, U_sz);
  fillRandom(d_bias,  Cout, 0.f, 0.01f);
  fillRandom(d_scale, Cout, 0.9f, 1.1f);

#pragma omp target enter data \
  map(to: d_input[0:input_sz], d_V[0:V_sz], d_U[0:U_sz], d_bias[0:Cout], d_scale[0:Cout]) \
  map(alloc: d_M[0:M_sz], d_output[0:out_sz])

  // warm-up
  winograd_BtdB(d_input, d_V, nTH, nTW, Cin);
  winograd_gemm(d_V, d_U, d_M, nTiles, Cin, Cout);
  winograd_AtIA(d_M, d_bias, d_scale, d_output, nTiles, Cout);

  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; ++r) {
    winograd_BtdB(d_input, d_V, nTH, nTW, Cin);
    winograd_gemm(d_V, d_U, d_M, nTiles, Cin, Cout);
    winograd_AtIA(d_M, d_bias, d_scale, d_output, nTiles, Cout);
  }
  auto t1 = std::chrono::steady_clock::now();
  double total = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-3;

#pragma omp target exit data \
  map(delete: d_input[0:input_sz], d_V[0:V_sz], d_U[0:U_sz], d_bias[0:Cout], d_scale[0:Cout], \
              d_M[0:M_sz], d_output[0:out_sz])
  delete[] d_input; delete[] d_V; delete[] d_U; delete[] d_M;
  delete[] d_bias; delete[] d_scale; delete[] d_output;
  return {total / repeat, total / repeat};
}

static KernelResult kernel_conv1x1(int repeat, int spatial, int Cin, int Cout) {
  float* d_A    = new float[spatial * Cin];
  float* d_W    = new float[Cin * Cout];
  float* d_bias = new float[Cout];
  float* d_scale= new float[Cout];
  float* d_C    = new float[spatial * Cout];

  fillRandom(d_A,    spatial * Cin);
  fillRandom(d_W,    Cin * Cout);
  fillRandom(d_bias, Cout, 0.f, 0.01f);
  fillRandom(d_scale,Cout, 0.9f, 1.1f);

#pragma omp target enter data \
  map(to: d_A[0:spatial*Cin], d_W[0:Cin*Cout], d_bias[0:Cout], d_scale[0:Cout]) \
  map(alloc: d_C[0:spatial*Cout])

  conv1x1(d_A, d_W, d_bias, d_scale, d_C, spatial, Cin, Cout);

  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; ++r)
    conv1x1(d_A, d_W, d_bias, d_scale, d_C, spatial, Cin, Cout);
  auto t1 = std::chrono::steady_clock::now();
  double total = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-3;

#pragma omp target exit data \
  map(delete: d_A[0:spatial*Cin], d_W[0:Cin*Cout], d_bias[0:Cout], d_scale[0:Cout], \
              d_C[0:spatial*Cout])
  delete[] d_A; delete[] d_W; delete[] d_bias; delete[] d_scale; delete[] d_C;
  return {total / repeat, total / repeat};
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <mode> <repeat (>2)>\n", argv[0]);
    return 1;
  }
  const int mode   = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  double time_total = 0, ktime_total = 0;
  const int effective = (repeat > 2) ? repeat - 2 : 1;

  for (int i = 0; i < repeat; ++i) {
    KernelResult r = {0, 0};
    switch (mode) {
      case 0: r = kernel_winograd(1, 128, 128); break;
      case 1: r = kernel_winograd(1, 256, 256); break;
      case 2: r = kernel_conv1x1(1, 14*14, 512, 128); break;
      case 3: r = kernel_conv1x1(1, 14*14, 128, 512); break;
      case 4: r = kernel_conv1x1(1, 7*7, 1024, 256); break;
      case 5: r = kernel_conv1x1(1, 7*7, 256, 1024); break;
      default: printf("Unknown mode %d\n", mode); return 1;
    }
    if (i > 1) {
      time_total  += r.time_us;
      ktime_total += r.ktime_us;
    }
  }

  printf("Case %d: Average device offload time: [%lf us]\n", mode, time_total / effective);
  printf("        Average kernel time: [%lf us]\n", ktime_total / effective);
  return 0;
}
