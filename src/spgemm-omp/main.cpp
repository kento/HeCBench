// spgemm – OpenMP target port of spgemm-kokkos
// Sparse matrix (COO/CSR) * Dense matrix → Dense matrix (SpMM)

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

static void init_coo(int *row_indices, float *values, int *col_indices,
                     float *matrix, int num_rows, int num_cols, int /*nnz*/) {
  int tmp = 0;
  for (int i = 0; i < num_rows; i++)
    for (int j = 0; j < num_cols; j++)
      if (matrix[i*num_cols+j] != 0.0f) {
        values[tmp] = matrix[i*num_cols+j];
        row_indices[tmp] = i;
        col_indices[tmp] = j;
        tmp++;
      }
}

static void init_csr(int *row_offsets, float *values, int *col_indices,
                     float *matrix, int num_rows, int num_cols, int nnz) {
  row_offsets[0] = 0;
  int *cnts = (int*)calloc(num_rows, sizeof(int));
  int tmp = 0;
  for (int i = 0; i < num_rows; i++)
    for (int j = 0; j < num_cols; j++)
      if (matrix[i*num_cols+j] != 0.0f) {
        values[tmp] = matrix[i*num_cols+j];
        col_indices[tmp] = j;
        tmp++; cnts[i]++;
      }
  for (int i = 1; i <= num_rows; i++) row_offsets[i] = row_offsets[i-1] + cnts[i-1];
  free(cnts);
}

static void gemm(float *A, float *B, float *C,
                 int A_num_cols, int C_num_rows, int C_num_cols) {
  for (int i = 0; i < C_num_rows; i++)
    for (int j = 0; j < C_num_cols; j++) {
      double s = 0;
      for (int k = 0; k < A_num_cols; k++) s += A[i*A_num_cols+k] * B[k*C_num_cols+j];
      C[i*C_num_cols+j] = (float)s;
    }
}

// COO SpMM: C(m,n) = A(m,k)[COO] * B(k,n)
void spmm_coo(int *d_rows, int *d_cols, float *d_vals, int nnz,
              float *d_B, float *d_C, int n_cols_C) {
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < nnz; i++) {
    int r = d_rows[i], kk = d_cols[i];
    float v = d_vals[i];
    for (int j = 0; j < n_cols_C; j++) {
      #pragma omp atomic
      d_C[r * n_cols_C + j] += v * d_B[kk * n_cols_C + j];
    }
  }
}

// CSR SpMM: C(m,n) = A(m,k)[CSR] * B(k,n)
void spmm_csr(int *d_offsets, int *d_cols, float *d_vals, int num_rows,
              float *d_B, float *d_C, int n_cols_C) {
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int r = 0; r < num_rows; r++) {
    for (int idx = d_offsets[r]; idx < d_offsets[r+1]; idx++) {
      int kk = d_cols[idx];
      float v = d_vals[idx];
      for (int j = 0; j < n_cols_C; j++)
        d_C[r * n_cols_C + j] += v * d_B[kk * n_cols_C + j];
    }
  }
}

int COO(int m, int k, int n, int a_nnz, int repeat, int verify) {
  float *hA = (float*)malloc(m * k * sizeof(float));
  float *hB = (float*)malloc(k * n * sizeof(float));
  float *hA_values = (float*)malloc(a_nnz * sizeof(float));
  int *hA_rows = (int*)malloc(a_nnz * sizeof(int));
  int *hA_cols = (int*)malloc(a_nnz * sizeof(int));

  init_matrix(hA, m, k, a_nnz);
  init_coo(hA_rows, hA_values, hA_cols, hA, m, k, a_nnz);
  init_matrix(hB, k, n, k * n);

  float *d_C = (float*)calloc(m * n, sizeof(float));

  #pragma omp target enter data \
    map(to: hA_rows[0:a_nnz], hA_cols[0:a_nnz], hA_values[0:a_nnz], hB[0:k*n]) \
    map(alloc: d_C[0:m*n])

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int j = 0; j < m*n; j++) d_C[j] = 0.0f;
    spmm_coo(hA_rows, hA_cols, hA_values, a_nnz, hB, d_C, n);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SPGEMM (COO) compute: %f (us)\n", (time * 1e-3f) / repeat);

  if (verify) {
    printf("Computing the reference SPGEMM results..\n");
    float *hC_ref = (float*)malloc(m * n * sizeof(float));
    gemm(hA, hB, hC_ref, k, m, n);
    #pragma omp target update from(d_C[0:m*n])
    int correct = 1;
    for (int i = 0; i < m*n; i++) if (fabsf(d_C[i] - hC_ref[i]) > 1e-2f) { correct = 0; break; }
    printf("%s\n", correct ? "spgemm_example test PASSED" : "spgemm_example test FAILED: wrong result");
    free(hC_ref);
  }

  #pragma omp target exit data map(delete: hA_rows[0:a_nnz], hA_cols[0:a_nnz], \
                                           hA_values[0:a_nnz], hB[0:k*n], d_C[0:m*n])
  free(hA); free(hB); free(hA_values); free(hA_rows); free(hA_cols); free(d_C);
  return 0;
}

int CSR(int m, int k, int n, int a_nnz, int repeat, int verify) {
  float *hA = (float*)malloc(m * k * sizeof(float));
  float *hB = (float*)malloc(k * n * sizeof(float));
  float *hA_values = (float*)malloc(a_nnz * sizeof(float));
  int *hA_cols = (int*)malloc(a_nnz * sizeof(int));
  int *hA_offsets = (int*)malloc((m+1) * sizeof(int));

  init_matrix(hA, m, k, a_nnz);
  init_csr(hA_offsets, hA_values, hA_cols, hA, m, k, a_nnz);
  init_matrix(hB, k, n, k * n);

  float *d_C = (float*)calloc(m * n, sizeof(float));

  #pragma omp target enter data \
    map(to: hA_offsets[0:m+1], hA_cols[0:a_nnz], hA_values[0:a_nnz], hB[0:k*n]) \
    map(alloc: d_C[0:m*n])

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int j = 0; j < m*n; j++) d_C[j] = 0.0f;
    spmm_csr(hA_offsets, hA_cols, hA_values, m, hB, d_C, n);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SPGEMM (CSR) compute: %f (us)\n", (time * 1e-3f) / repeat);

  if (verify) {
    printf("Computing the reference SPGEMM results..\n");
    float *hC_ref = (float*)malloc(m * n * sizeof(float));
    gemm(hA, hB, hC_ref, k, m, n);
    #pragma omp target update from(d_C[0:m*n])
    int correct = 1;
    for (int i = 0; i < m*n; i++) if (fabsf(d_C[i] - hC_ref[i]) > 1e-2f) { correct = 0; break; }
    printf("%s\n", correct ? "spgemm_example test PASSED" : "spgemm_example test FAILED: wrong result");
    free(hC_ref);
  }

  #pragma omp target exit data map(delete: hA_offsets[0:m+1], hA_cols[0:a_nnz], \
                                           hA_values[0:a_nnz], hB[0:k*n], d_C[0:m*n])
  free(hA); free(hB); free(hA_values); free(hA_cols); free(hA_offsets); free(d_C);
  return 0;
}

int main(int argc, char *argv[]) {
  if (argc != 7) {
    printf("Sparse matrix-dense matrix multiplication (SpMM)\n");
    printf("Usage %s <M> <K> <N> <A_nnz> <repeat> <verify>\n", argv[0]);
    return 1;
  }
  int m = atoi(argv[1]), k = atoi(argv[2]), n = atoi(argv[3]);
  int a_nnz = atoi(argv[4]), repeat = atoi(argv[5]), verify = atoi(argv[6]);

  COO(m, k, n, a_nnz, repeat, verify);
  CSR(m, k, n, a_nnz, repeat, verify);
  return 0;
}
