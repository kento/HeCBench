// OpenMP target offloading port of gels-kokkos (batched least-squares via QR)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <omp.h>

#pragma omp declare target
static void qr_solve(double* A, int m, int n, int lda, double* x)
{
  double R[5][5] = {};
  double Q[25];

  for (int c = 0; c < n; c++)
    for (int r = 0; r < m; r++)
      Q[r + c*m] = A[r + c*lda];

  // Modified Gram-Schmidt
  for (int j = 0; j < n; j++) {
    double norm = 0.0;
    for (int r = 0; r < m; r++) norm += Q[r + j*m] * Q[r + j*m];
    norm = sqrt(norm);
    R[j][j] = norm;
    if (norm < 1e-12) continue;
    double inv = 1.0 / norm;
    for (int r = 0; r < m; r++) Q[r + j*m] *= inv;
    for (int k = j+1; k < n; k++) {
      double dot = 0.0;
      for (int r = 0; r < m; r++) dot += Q[r + j*m] * Q[r + k*m];
      R[j][k] = dot;
      for (int r = 0; r < m; r++) Q[r + k*m] -= dot * Q[r + j*m];
    }
  }

  // Qt * b
  double Qtb[5] = {};
  for (int j = 0; j < n; j++)
    for (int r = 0; r < m; r++)
      Qtb[j] += Q[r + j*m] * x[r];

  // Back-substitution
  for (int j = n-1; j >= 0; j--) {
    double s = Qtb[j];
    for (int k = j+1; k < n; k++) s -= R[j][k] * x[k];
    x[j] = (R[j][j] != 0.0) ? s / R[j][j] : 0.0;
  }
}
#pragma omp end declare target

static int run_gels(int repeat)
{
  const int m = 5, n = 5, nrhs = 1, lda = m;
  const int stride_a = n * lda, stride_b = nrhs * m, batch_size = 2;

  double A_host[] = {
     1.0,  1.0,  1.0,  1.0,  1.0,
     0.0,  0.2,  0.6,  1.0,  1.8,
     0.0, -0.4, -0.2, -1.0, -0.6,
     0.0, -0.4,  0.4,  0.6,  0.2,
     0.0, -0.8, -1.2, -0.8, -0.6,

     0.2,  0.4,  0.4,  0.8,  0.0,
    -0.4,  0.2, -0.8,  0.4,  0.0,
    -0.4,  0.8,  0.2, -0.4,  0.0,
    -0.8, -0.4,  0.4,  0.2,  0.0,
     0.0,  0.0,  0.0,  0.0,  1.0
  };

  double B_host[] = {
     5.0,  3.6, -2.2,  0.8, -3.4,
     1.8, -0.6,  0.2, -0.6,  1.0
  };

  const double X_expected[] = {1,1,1,1,1, 1,1,1,1,1};

  int A_sz = stride_a * batch_size;
  int B_sz = stride_b * batch_size;

  double* d_A = (double*)malloc(A_sz * sizeof(double));
  double* d_B = (double*)malloc(B_sz * sizeof(double));

  for (int i = 0; i < A_sz; i++) d_A[i] = A_host[i];
  for (int i = 0; i < B_sz; i++) d_B[i] = B_host[i];

  #pragma omp target enter data map(tofrom: d_A[0:A_sz], d_B[0:B_sz])

  auto t0 = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    #pragma omp target teams distribute parallel for num_teams(1) thread_limit(1)
    for (int b = 0; b < batch_size; b++) {
      double* Ap = &d_A[b * stride_a];
      double* bp = &d_B[b * stride_b];
      qr_solve(Ap, m, n, lda, bp);
    }
    // Restore A and B for next iteration
    if (r < repeat - 1) {
      #pragma omp target update from(d_A[0:A_sz], d_B[0:B_sz])
      for (int i = 0; i < A_sz; i++) d_A[i] = A_host[i];
      for (int i = 0; i < B_sz; i++) d_B[i] = B_host[i];
      #pragma omp target update to(d_A[0:A_sz], d_B[0:B_sz])
    }
  }

  auto t1 = std::chrono::steady_clock::now();
  double us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-3;
  printf("Average kernel execution time : %f (us)\n", us / repeat);

  #pragma omp target update from(d_B[0:B_sz])

  const double bound = 1e-8;
  bool passed = true;
  printf("Results:\n");
  for (int b = 0; b < batch_size; b++) {
    for (int j = 0; j < n; j++) {
      double v = d_B[b * stride_b + j];
      printf("%6.2f ", v);
      if (fabs(v - X_expected[b * n + j]) > bound) passed = false;
    }
    printf("\n");
  }

  if (passed) printf("Calculations successfully finished\n");
  else        printf("ERROR: results mismatch!\n");

  #pragma omp target exit data map(delete: d_A[0:A_sz], d_B[0:B_sz])
  free(d_A); free(d_B);

  return passed ? 0 : 1;
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]); return 1;
  }
  int repeat = atoi(argv[1]);

  std::cout << "\n#############################################\n";
  std::cout << "# Batched GELS OMP port (real double)\n";
  std::cout << "#############################################\n\n";

  return run_gels(repeat);
}
