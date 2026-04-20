// OpenMP target offloading port of rayleighBenardConvection benchmark.
// Spectral finite-difference Rayleigh-Benard convection with RK3 time integration.

#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

static constexpr int   M          = 500;
static constexpr int   N          = 1000;
static constexpr float RA         = 1e8f;
static constexpr int   XF         = 2;
static constexpr int   STARTSTEP  = 0;
static constexpr float PI         = 3.14159265358979323846f;
static constexpr float DX         = 1.f / (M - 1.f);
static constexpr float DX2        = DX * DX;
static constexpr float OMEGACOEFF = -((DX2) * (DX2)) * RA;
static constexpr float DT_START   = 0.000000000000005f;
static constexpr float FRAMESIZE  = DX2 / 4.f;

// ---------- Element-wise helpers ----------

static void elemMultOmega(float* A, const float* B, int size) {
#pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < size; i++) A[i] *= B[i];
}

static void elemMultT(float* A, const float* B, int size) {
#pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < size; i++) A[i] *= B[i];
}

static void saxpy_d(float alpha, const float* x, float* y, int n) {
#pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < n; i++) y[i] += alpha * x[i];
}

static void scopy_d(const float* src, float* dst, int n) {
#pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < n; i++) dst[i] = src[i];
}

static void sscal_d(float alpha, float* y, int n) {
#pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < n; i++) y[i] *= alpha;
}

static void saxpy_off(float alpha, const float* x, int xoff,
                      float* y, int yoff, int n) {
#pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < n; i++) y[yoff + i] += alpha * x[xoff + i];
}

static void scopy_off(const float* src, int soff, float* dst, int doff, int n) {
#pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < n; i++) dst[doff + i] = src[soff + i];
}

static void sscal_off(float alpha, float* y, int off, int n) {
#pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < n; i++) y[off + i] *= alpha;
}

static void fill_d(float* y, float val, int n) {
#pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < n; i++) y[i] = val;
}

// GEMM: C = alpha*A*B + beta*C  (all row-major, A[m x k], B[k x n], C[m x n])
static void sgemm(int m, int n, int k,
                  float alpha,
                  const float* A, int A_off,
                  const float* B, int B_off,
                  float beta,
                  float* C, int C_off) {
#pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
  for (int row = 0; row < m; row++) {
    for (int col = 0; col < n; col++) {
      float sum = 0.f;
      for (int p = 0; p < k; ++p)
        sum += A[A_off + row * k + p] * B[B_off + p * n + col];
      C[C_off + row * n + col] = alpha * sum + beta * C[C_off + row * n + col];
    }
  }
}

// Dx: first derivative in x-direction
static void Dx(const float* f, int is_psi, float* y) {
  const float alpha2 = 1.f / (2.f * DX);
#pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
  for (int row = 0; row < M-2; row++) {
    for (int col = 1; col < N-1; col++)
      y[row * N + col] = alpha2 * (f[row * N + col + 1] - f[row * N + col - 1]);
  }
#pragma omp target teams distribute parallel for thread_limit(256)
  for (int row = 0; row < M-2; row++) {
    float bc = is_psi ? 0.f : 1.f;
    y[row * N]     = alpha2 * (f[row * N + 1] - bc);
    y[row * N + N - 1] = alpha2 * (bc - f[row * N + N - 2]);
  }
}

// Dz: first derivative in z-direction
static void Dz(const float* f, int is_psi, float* y) {
  const float alpha2 = 1.f / (2.f * DX);
#pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
  for (int row = 1; row < M-3; row++) {
    for (int col = 0; col < N; col++)
      y[row * N + col] = alpha2 * (f[(row+1)*N + col] - f[(row-1)*N + col]);
  }
#pragma omp target teams distribute parallel for thread_limit(256)
  for (int col = 0; col < N; col++) {
    float bc = is_psi ? 0.f : 1.f;
    y[col]         = alpha2 * (f[N + col] - bc);
    y[(M-3)*N+col] = alpha2 * (0.f - f[(M-4)*N + col]);
  }
}

// Dxx: second x-derivative
static void Dxx(const float* f, float* y) {
  const float c = 1.f / DX2;
#pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
  for (int row = 0; row < M-2; row++) {
    for (int col = 0; col < N; col++) {
      float fm = (col > 0)   ? f[row * N + col - 1] : 0.f;
      float f0 = f[row * N + col];
      float fp = (col < N-1) ? f[row * N + col + 1] : 0.f;
      y[row * N + col] = c * (fm - 2.f*f0 + fp);
    }
  }
}

// Dzz: second z-derivative
static void Dzz(const float* f, float* y) {
  const float c = 1.f / DX2;
#pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
  for (int row = 0; row < M-2; row++) {
    for (int col = 0; col < N; col++) {
      float fm = (row > 0)   ? f[(row-1)*N + col] : 0.f;
      float f0 = f[row * N + col];
      float fp = (row < M-3) ? f[(row+1)*N + col] : 0.f;
      y[row * N + col] = c * (fm - 2.f*f0 + fp);
    }
  }
}

static void updatedt(const float* u, const float* v, float& dt) {
  const int n = (M-2) * N;
  float maxU = 0.f, maxV = 0.f;
#pragma omp target teams distribute parallel for reduction(max:maxU) thread_limit(256)
  for (int i = 0; i < n; i++) {
    float av = u[i] < 0 ? -u[i] : u[i];
    if (av > maxU) maxU = av;
  }
#pragma omp target teams distribute parallel for reduction(max:maxV) thread_limit(256)
  for (int i = 0; i < n; i++) {
    float av = v[i] < 0 ? -v[i] : v[i];
    if (av > maxV) maxV = av;
  }
  float maxuv = maxU > maxV ? maxU : maxV;
  if (maxuv > 0.f)
    dt = std::min(DX / maxuv, DX2 / 4.f);
}

static float nusseltCompute(const float* T, const float* trnu,
                             float* nutop_buf) {
  // Copy last 3 rows of T into nutop_buf
  scopy_off(T, (M-5)*N, nutop_buf, 0,   N);
  scopy_off(T, (M-4)*N, nutop_buf, N,   N);
  scopy_off(T, (M-3)*N, nutop_buf, 2*N, N);
#pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < N; i++) nutop_buf[3*N + i] = 0.f;

  // Bring to host for final computation
  std::vector<float> h_nutop(4*N), h_trnu(N);
#pragma omp target update from(nutop_buf[0:4*N], trnu[0:N])
  for (int j = 0; j < 4*N; j++) h_nutop[j] = nutop_buf[j];
  for (int j = 0; j < N;   j++) h_trnu[j]  = trnu[j];

  float result = 0.f;
  for (int j = 0; j < N; ++j) {
    float dT = (-(2.f/3.f) * h_nutop[j]
                + 3.f * h_nutop[N + j]
                - 6.f * h_nutop[2*N + j]
                + (11.f/3.f) * h_nutop[3*N + j]) / (2.f * DX);
    result += dT * h_trnu[j];
  }
  return result / (-XF);
}

static void G(float* f, float* Tbuff, float* DxT,
              float* y, float* u, float* v,
              float* psi, float* omega,
              const float* dsc, const float* dsr, const float* ei,
              float& dt, float* output, bool compute_velocity) {
  const int tSize    = (M-2) * N;
  const int innerSize = (M-2) * (N-2);
  const float alpha = 1.f, beta = 0.f;

  scopy_d(f, DxT, tSize);
  Dx(f, 0, DxT);

  // Extract interior columns of DxT into omega
#pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
  for (int i = 0; i < M-2; i++)
    for (int j = 0; j < N-2; j++)
      omega[i*(N-2) + j] = DxT[i*N + j + 1];

  sgemm(N-2, M-2, M-2, alpha, omega, 0, dsc, 0, beta, Tbuff, 0);
  scopy_off(Tbuff, 0, omega, 0, innerSize);
  sgemm(N-2, M-2, N-2, alpha, dsr, 0, omega, 0, beta, Tbuff, 0);
  scopy_off(Tbuff, 0, omega, 0, innerSize);
  elemMultOmega(omega, ei, innerSize);
  sgemm(N-2, M-2, M-2, alpha, omega, 0, dsc, 0, beta, Tbuff, 0);
  scopy_off(Tbuff, 0, omega, 0, innerSize);
  sgemm(N-2, M-2, N-2, alpha, dsr, 0, omega, 0, beta, Tbuff, 0);
  scopy_off(Tbuff, 0, omega, 0, innerSize);
  sscal_d(OMEGACOEFF, omega, innerSize);

#pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
  for (int i = 0; i < M-2; i++)
    for (int j = 0; j < N-2; j++)
      psi[i*N + j + 1] = omega[i*(N-2) + j];

  scopy_d(f, u, tSize);
  Dz(psi, 1, u);

  scopy_d(f, v, tSize);
  Dx(psi, 1, v);
  sscal_d(-1.f, v, tSize);

  if (compute_velocity) updatedt(u, v, dt);

  scopy_d(f, y, tSize);
  Dxx(f, y);

  scopy_d(f, Tbuff, tSize);
  Dzz(f, Tbuff);
  saxpy_d(1.f, Tbuff, y, tSize);

  elemMultT(u, DxT, tSize);
  saxpy_d(1.f, u, y, tSize);

  scopy_d(f, u, tSize);
  Dz(f, 0, u);
  elemMultT(u, v, tSize);
  saxpy_d(1.f, u, y, tSize);

  scopy_d(y, output, tSize);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <timesteps>\n", argv[0]);
    return 1;
  }
  const int ENDSTEP = atoi(argv[1]);
  printf("M = %d. N = %d. DX = %E. Ra = %E.\n", M, N, DX, RA);
  printf("\nInitialization\n");

  const int tSize    = (M-2) * N;
  const int innerSz  = (M-2) * (N-2);
  const int dscSz    = (M-2) * (M-2);
  const int dsrSz    = (N-2) * (N-2);

  float* d_T      = (float*)malloc(tSize    * sizeof(float));
  float* d_Tbuff  = (float*)malloc(tSize    * sizeof(float));
  float* d_DxT    = (float*)malloc(tSize    * sizeof(float));
  float* d_y      = (float*)malloc(tSize    * sizeof(float));
  float* d_u      = (float*)malloc(tSize    * sizeof(float));
  float* d_v      = (float*)malloc(tSize    * sizeof(float));
  float* d_psi    = (float*)malloc(tSize    * sizeof(float));
  float* d_omega  = (float*)malloc(innerSz  * sizeof(float));
  float* d_dsc    = (float*)malloc(dscSz    * sizeof(float));
  float* d_dsr    = (float*)malloc(dsrSz    * sizeof(float));
  float* d_ei     = (float*)malloc(innerSz  * sizeof(float));
  float* d_xrk3   = (float*)malloc(tSize    * sizeof(float));
  float* d_yrk3   = (float*)malloc(tSize    * sizeof(float));
  float* d_zrk3   = (float*)malloc(tSize    * sizeof(float));
  float* d_temp   = (float*)malloc(tSize    * sizeof(float));
  float* d_trnu   = (float*)malloc(N        * sizeof(float));
  float* d_nutop  = (float*)malloc(4*N      * sizeof(float));

  // Host initialization
  std::vector<float> h_X(N), h_Z(M-2), h_T(tSize);
  std::vector<float> h_dsc(dscSz), h_dsr(dsrSz);
  std::vector<float> h_lambda(M-2), h_mu(N-2), h_ei(innerSz);
  std::vector<float> h_trnu(N);

  for (int i = 0; i < N; ++i)   h_X[i] = (i * XF + 0.f) / (N - 1.f);
  for (int i = 1; i < M-1; ++i) h_Z[i-1] = (i + 0.f) / (M - 1.f);

  for (int i = 0; i < M-2; ++i)
    for (int j = 0; j < N; ++j)
      h_T[i*N + j] = 1 - h_Z[i] + 0.01f * std::sin(PI * h_Z[i]) * std::cos((PI/XF) * h_X[j]);

  for (int i = 0; i < M-2; ++i)
    for (int j = 0; j < M-2; ++j)
      h_dsc[(M-2)*i + j] = std::sqrt(2.f/(M-1.f)) * std::sin((i+1.f)*(j+1.f)*PI/(M-1.f));

  for (int i = 0; i < N-2; ++i)
    for (int j = 0; j < N-2; ++j)
      h_dsr[(N-2)*i + j] = std::sqrt(2.f/(N-1.f)) * std::sin((i+1.f)*(j+1.f)*PI/(N-1.f));

  for (int i = 0; i < M-2; ++i) h_lambda[i] = 2.f*std::cos((i+1.f)*PI/(M-1.f)) - 2.f;
  for (int i = 0; i < N-2; ++i) h_mu[i]     = 2.f*std::cos((i+1.f)*PI/(N-1.f)) - 2.f;

  for (int i = 0; i < M-2; ++i)
    for (int j = 0; j < N-2; ++j) {
      float v = h_lambda[i] + h_mu[j];
      h_ei[(N-2)*i + j] = 1.f / (v * v);
    }

  for (int i = 1; i < N-1; ++i) h_trnu[i] = DX;
  h_trnu[0] = DX / 2.f;
  h_trnu[N-1] = DX / 2.f;

  // Copy init data to host buffers
  for (int i = 0; i < tSize;   i++) d_T[i]    = h_T[i];
  for (int i = 0; i < dscSz;   i++) d_dsc[i]  = h_dsc[i];
  for (int i = 0; i < dsrSz;   i++) d_dsr[i]  = h_dsr[i];
  for (int i = 0; i < innerSz; i++) d_ei[i]   = h_ei[i];
  for (int i = 0; i < N;       i++) d_trnu[i] = h_trnu[i];
  for (int i = 0; i < tSize;   i++) { d_psi[i]=0.f; d_u[i]=0.f; d_v[i]=0.f; d_y[i]=0.f; }
  for (int i = 0; i < innerSz; i++) d_omega[i] = 0.f;
  for (int i = 0; i < tSize;   i++) {
    d_xrk3[i] = d_T[i]; d_yrk3[i] = d_T[i]; d_zrk3[i] = d_T[i];
    d_Tbuff[i] = d_T[i]; d_DxT[i] = d_T[i]; d_temp[i] = d_T[i];
  }

#pragma omp target enter data \
  map(to: d_T[0:tSize], d_dsc[0:dscSz], d_dsr[0:dsrSz], d_ei[0:innerSz], d_trnu[0:N]) \
  map(alloc: d_Tbuff[0:tSize], d_DxT[0:tSize], d_y[0:tSize], d_u[0:tSize], d_v[0:tSize], \
             d_psi[0:tSize], d_omega[0:innerSz], d_xrk3[0:tSize], d_yrk3[0:tSize], \
             d_zrk3[0:tSize], d_temp[0:tSize], d_nutop[0:4*N])

  // Initialize arrays on device
  scopy_d(d_T, d_xrk3,  tSize);
  scopy_d(d_T, d_yrk3,  tSize);
  scopy_d(d_T, d_zrk3,  tSize);
  scopy_d(d_T, d_Tbuff, tSize);
  scopy_d(d_T, d_DxT,   tSize);
  scopy_d(d_T, d_temp,  tSize);
  fill_d(d_psi,   0.f, tSize);
  fill_d(d_u,     0.f, tSize);
  fill_d(d_v,     0.f, tSize);
  fill_d(d_y,     0.f, tSize);
  fill_d(d_omega, 0.f, innerSz);

  printf("Begin computation:\n");

  float dt     = DT_START;
  float frames = 0.f;

  auto t_start = std::chrono::steady_clock::now();

  for (int c = STARTSTEP; c < ENDSTEP; ++c) {
    G(d_T, d_Tbuff, d_DxT, d_y, d_u, d_v,
      d_psi, d_omega, d_dsc, d_dsr, d_ei, dt, d_xrk3, true);

    scopy_d(d_T, d_temp, tSize);
    saxpy_d(dt / 3.f, d_xrk3, d_temp, tSize);

    G(d_temp, d_Tbuff, d_DxT, d_y, d_u, d_v,
      d_psi, d_omega, d_dsc, d_dsr, d_ei, dt, d_yrk3, false);

    scopy_d(d_T, d_temp, tSize);
    saxpy_d(2.f * dt / 3.f, d_yrk3, d_temp, tSize);

    G(d_temp, d_Tbuff, d_DxT, d_y, d_u, d_v,
      d_psi, d_omega, d_dsc, d_dsr, d_ei, dt, d_zrk3, false);

    saxpy_d(dt / 4.f,       d_xrk3, d_T, tSize);
    saxpy_d(3.f * dt / 4.f, d_zrk3, d_T, tSize);

    frames += dt;
    if (frames > FRAMESIZE) {
      float nu = nusseltCompute(d_T, d_trnu, d_nutop);
      printf("Nusselt number: %.1f\n", nu);
      frames = 0.f;
    }
  }

  auto t_end = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() * 1e-9;
  printf("Total compute time: %f (s)\n", elapsed);
  if (ENDSTEP > 0)
    printf("Average compute time per step: %f (s)\n", elapsed / ENDSTEP);

#pragma omp target exit data \
  map(delete: d_T[0:tSize], d_dsc[0:dscSz], d_dsr[0:dsrSz], d_ei[0:innerSz], d_trnu[0:N], \
              d_Tbuff[0:tSize], d_DxT[0:tSize], d_y[0:tSize], d_u[0:tSize], d_v[0:tSize], \
              d_psi[0:tSize], d_omega[0:innerSz], d_xrk3[0:tSize], d_yrk3[0:tSize], \
              d_zrk3[0:tSize], d_temp[0:tSize], d_nutop[0:4*N])

  free(d_T); free(d_Tbuff); free(d_DxT); free(d_y); free(d_u); free(d_v);
  free(d_psi); free(d_omega); free(d_dsc); free(d_dsr); free(d_ei);
  free(d_xrk3); free(d_yrk3); free(d_zrk3); free(d_temp); free(d_trnu); free(d_nutop);
  return 0;
}
