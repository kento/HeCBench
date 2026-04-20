/*
 * OpenMP target offloading port of bicgstab.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cmath>
#include <vector>
#include <omp.h>

int main(int argc, char* argv[]) {
  const int m       = 1000;
  const int maxIter = 300;
  const double tol  = 1e-10;

  std::vector<double> h_A(m * m, 0.0);
  std::vector<double> h_b(m, 1.0);
  for (int i = 0; i < m; i++) {
    h_A[i * m + i] = 4.0;
    if (i > 0)     h_A[i * m + (i-1)] = -1.0;
    if (i < m - 1) h_A[i * m + (i+1)] = -1.0;
  }

  double* A  = (double*)malloc(m * m * sizeof(double));
  double* b  = (double*)malloc(m * sizeof(double));
  double* x  = (double*)calloc(m, sizeof(double));
  double* r  = (double*)malloc(m * sizeof(double));
  double* r0 = (double*)malloc(m * sizeof(double));
  double* p  = (double*)malloc(m * sizeof(double));
  double* v  = (double*)malloc(m * sizeof(double));
  double* s  = (double*)malloc(m * sizeof(double));
  double* t  = (double*)malloc(m * sizeof(double));

  memcpy(A, h_A.data(), m * m * sizeof(double));
  memcpy(b, h_b.data(), m * sizeof(double));

  #pragma omp target enter data map(to: A[0:m*m], b[0:m]) \
    map(alloc: x[0:m], r[0:m], r0[0:m], p[0:m], v[0:m], s[0:m], t[0:m])

  auto dot_f = [&](double* u, double* w) -> double {
    double res = 0.0;
    #pragma omp target teams distribute parallel for reduction(+:res) thread_limit(256)
    for (int i = 0; i < m; i++) res += u[i] * w[i];
    return res;
  };

  auto axpy_f = [&](double alpha, double* src, double* dst) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < m; i++) dst[i] += alpha * src[i];
  };

  auto copy_v = [&](double* src, double* dst) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < m; i++) dst[i] = src[i];
  };

  auto mv_f = [&](double* in, double* out) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < m; i++) {
      double sum = 0.0;
      for (int j = 0; j < m; j++) sum += A[i * m + j] * in[j];
      out[i] = sum;
    }
  };

  auto nrm2_f = [&](double* u) -> double {
    double res = 0.0;
    #pragma omp target teams distribute parallel for reduction(+:res) thread_limit(256)
    for (int i = 0; i < m; i++) res += u[i] * u[i];
    return sqrt(res);
  };

  // init on device
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < m; i++) x[i] = 0.0;

  auto start = std::chrono::steady_clock::now();

  copy_v(b, r);
  copy_v(r, r0);
  copy_v(r, p);

  double rho = dot_f(r0, r);
  double nrm_r0 = nrm2_f(r);
  double threshold = tol * nrm_r0;
  double nrm_r = nrm_r0;

  int iter = 0;
  double alpha = 1.0, omega = 1.0;
  for (iter = 1; iter <= maxIter && nrm_r > threshold; iter++) {
    mv_f(p, v);
    double rv = dot_f(r0, v);
    alpha = rho / rv;

    copy_v(r, s);
    axpy_f(-alpha, v, s);

    nrm_r = nrm2_f(s);
    if (nrm_r <= threshold) {
      axpy_f(alpha, p, x);
      break;
    }

    mv_f(s, t);
    omega = dot_f(t, s) / dot_f(t, t);

    axpy_f(alpha, p, x);
    axpy_f(omega, s, x);

    copy_v(s, r);
    axpy_f(-omega, t, r);
    nrm_r = nrm2_f(r);

    double rho_new = dot_f(r0, r);
    double beta = (rho_new / rho) * (alpha / omega);
    rho = rho_new;

    axpy_f(-omega, v, p);
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < m; i++) p[i] = r[i] + beta * p[i];
  }

  auto end = std::chrono::steady_clock::now();
  auto time_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-6;

  printf("BiCGStab converged in %d iterations, residual = %e\n", iter-1, nrm_r);
  printf("Elapsed time: %.3f ms\n", time_ms);

  #pragma omp target exit data map(delete: A[0:m*m], b[0:m], x[0:m], r[0:m], r0[0:m], \
                                           p[0:m], v[0:m], s[0:m], t[0:m])
  free(A); free(b); free(x); free(r); free(r0);
  free(p); free(v); free(s); free(t);
  return 0;
}
