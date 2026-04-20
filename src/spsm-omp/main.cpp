// spsm – OpenMP target port of spsm-cuda
// Sparse triangular matrix solve: A * C = B  (lower triangular CSR, row-parallel solve)

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

static void init_matrix(float *matrix, int num_rows, int num_cols, int nnz) {
  int n = num_rows * num_cols;
  float *d = (float*)malloc(n * sizeof(float));
  srand(123);
  for (int i = 0; i < n; i++) d[i] = (float)i;
  for (int i = n; i > 0; i--) {
    int a = i-1, b = rand() % i;
    if (a != b) { auto t = d[a]; d[a] = d[b]; d[b] = t; }
  }
  srand48(123);
  for (int i = 0; i < num_rows; i++)
    for (int j = 0; j < num_cols; j++)
      matrix[i*num_cols+j] = (d[i*num_cols+j] >= nnz) ? 0.0f : (float)(drand48()+1);
  free(d);
}

static void init_csr(int *row_indices, float *values, int *col_indices,
                     float *matrix, int num_rows, int num_cols, int nnz) {
  row_indices[0] = 0;
  row_indices[num_rows] = nnz;
  int *nze = (int*)calloc(num_rows, sizeof(int));
  int tmp = 0;
  for (int i = 0; i < num_rows; i++)
    for (int j = 0; j < num_cols; j++)
      if (matrix[i*num_cols+j] != 0.0f) {
        values[tmp] = matrix[i*num_cols+j]; col_indices[tmp] = j;
        tmp++; nze[i]++;
      }
  for (int i = 1; i < num_rows; i++) row_indices[i] = row_indices[i-1] + nze[i-1];
  free(nze);
}

// Reference: lower triangular forward substitution  A * C = B
static void spsm_ref(float *A, float *C, float *B, int A_num_rows, int C_num_cols) {
  for (int i = 0; i < A_num_rows; i++)
    for (int j = 0; j < C_num_cols; j++) {
      double s = 0;
      for (int k = 0; k <= i; k++) s += A[i*A_num_rows+k] * C[k*C_num_cols+j];
      B[i*C_num_cols+j] = (float)s;
    }
}

// GPU forward substitution row-by-row (sequential across rows, parallel across RHS columns)
// Uses level-set parallelism: row i depends only on rows 0..i-1
// We handle this by doing the solve row-by-row, but each row can be parallelized over nrhs
static void spsm_device(int *dA_off, int *dA_col, float *dA_val,
                        float *dC, float *dX,
                        int m, int nrhs) {
  // Copy initial C to X
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < m * nrhs; i++) dX[i] = dC[i];

  // Forward substitution row by row
  for (int row = 0; row < m; row++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int j = 0; j < nrhs; j++) {
      float s = dX[row * nrhs + j];
      float diag = 1.0f;
      for (int idx = dA_off[row]; idx < dA_off[row+1]; idx++) {
        int col = dA_col[idx];
        if (col < row)       s -= dA_val[idx] * dX[col * nrhs + j];
        else if (col == row) diag = dA_val[idx];
      }
      dX[row * nrhs + j] = s / diag;
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc != 6) {
    printf("Solves a system of linear equations A * C = B (lower triangular CSR).\n");
    printf("Usage %s <M> <N> <A_nnz> <repeat> <verify>\n", argv[0]);
    return 1;
  }
  int m = atoi(argv[1]), n = atoi(argv[2]), a_nnz = atoi(argv[3]);
  int repeat = atoi(argv[4]), verify = atoi(argv[5]);

  int A_size = m * m, C_size = m * n, B_size = m * n;
  float *hA  = (float*)malloc(A_size * sizeof(float));
  float *hB  = (float*)malloc(B_size * sizeof(float));
  float *hC  = (float*)malloc(C_size * sizeof(float));
  float *hA_val = (float*)malloc(a_nnz * sizeof(float));
  int   *hA_col = (int*)  malloc(a_nnz * sizeof(int));
  int   *hA_off = (int*)  malloc((m+1) * sizeof(int));

  printf("Initializing host matrices..\n");
  init_matrix(hA, m, m, a_nnz);
  init_csr(hA_off, hA_val, hA_col, hA, m, m, a_nnz);
  init_matrix(hC, m, n, C_size);
  spsm_ref(hA, hC, hB, m, n);
  printf("Done\n");

  float *dX = (float*)malloc(B_size * sizeof(float));

  #pragma omp target enter data \
    map(to: hA_off[0:m+1], hA_col[0:a_nnz], hA_val[0:a_nnz], hC[0:C_size]) \
    map(alloc: dX[0:B_size])

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++)
    spsm_device(hA_off, hA_col, hA_val, hC, dX, m, n);
  auto end = std::chrono::steady_clock::now();
  printf("Average execution time of SpSM solve: %f (us)\n",
         (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() * 1e-3f) / repeat);

  if (verify) {
    printf("Checking results..\n");
    #pragma omp target update from(dX[0:B_size])
    // Re-compute B from device result
    float *hB2 = (float*)malloc(B_size * sizeof(float));
    spsm_ref(hA, dX, hB2, m, n);
    int correct = 1;
    for (int i = 0; i < m*n; i++) if (fabsf(hB[i]-hB2[i]) > 1e-2f) { correct=0; break; }
    printf("%s\n", correct ? "spsm_csr_example test PASSED" : "spsm_csr_example test FAILED: wrong result");
    free(hB2);
  }

  #pragma omp target exit data map(delete: hA_off[0:m+1], hA_col[0:a_nnz], hA_val[0:a_nnz], \
                                           hC[0:C_size], dX[0:B_size])
  free(hA); free(hB); free(hC); free(hA_val); free(hA_col); free(hA_off); free(dX);
  return 0;
}
