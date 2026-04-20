// sps2d – OpenMP target port of sps2d-kokkos
// Sparse CSR matrix → dense matrix conversion

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

static void init_matrix(float *matrix, int64_t num_rows, int64_t num_cols, int64_t nnz) {
  int64_t n = num_rows * num_cols;
  float *d = (float*)malloc(n * sizeof(float));
  srand(123);
  for (int64_t i = 0; i < n; i++) d[i] = (float)i;
  for (int64_t i = n; i > 0; i--) {
    int64_t a = i-1, b = rand() % i;
    if (a != b) { auto t = d[a]; d[a] = d[b]; d[b] = t; }
  }
  srand48(123);
  for (int64_t i = 0; i < num_rows; i++)
    for (int64_t j = 0; j < num_cols; j++)
      matrix[i*num_cols+j] = (d[i*num_cols+j] >= (float)nnz) ? 0.0f : (float)(drand48()+1);
  free(d);
}

static void init_csr(int64_t *row_offsets, float *values, int64_t *col_indices,
                     float *matrix, int64_t num_rows, int64_t num_cols, int64_t nnz) {
  row_offsets[0] = 0;
  int64_t *cnts = (int64_t*)calloc(num_rows, sizeof(int64_t));
  int64_t tmp = 0;
  for (int64_t i = 0; i < num_rows; i++)
    for (int64_t j = 0; j < num_cols; j++)
      if (matrix[i*num_cols+j] != 0.0f) {
        values[tmp] = matrix[i*num_cols+j];
        col_indices[tmp] = j;
        tmp++; cnts[i]++;
      }
  for (int64_t i = 1; i <= num_rows; i++) row_offsets[i] = row_offsets[i-1] + cnts[i-1];
  free(cnts);
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("Sparse CSR matrix to dense matrix conversion\n");
    printf("Usage %s <M> <N> <nnz> <repeat>\n", argv[0]);
    return 1;
  }
  int64_t m = atol(argv[1]);
  int64_t n = atol(argv[2]);
  int64_t h_nnz = atol(argv[3]);
  int repeat = atoi(argv[4]);

  float   *h_dense        = (float*)  malloc(m * n * sizeof(float));
  float   *h_dense_result = (float*)  malloc(m * n * sizeof(float));
  float   *h_csr_values   = (float*)  malloc(h_nnz * sizeof(float));
  int64_t *h_csr_columns  = (int64_t*)malloc(h_nnz * sizeof(int64_t));
  int64_t *h_csr_offsets  = (int64_t*)malloc((m+1) * sizeof(int64_t));

  printf("Initializing host matrices..\n");
  init_matrix(h_dense, m, n, h_nnz);
  init_csr(h_csr_offsets, h_csr_values, h_csr_columns, h_dense, m, n, h_nnz);

  float   *d_dense   = (float*)  calloc(m * n, sizeof(float));
  float   *d_csr_val = h_csr_values;
  int64_t *d_csr_col = h_csr_columns;
  int64_t *d_csr_off = h_csr_offsets;

  #pragma omp target enter data \
    map(alloc: d_dense[0:m*n]) \
    map(to: d_csr_off[0:m+1], d_csr_col[0:h_nnz], d_csr_val[0:h_nnz])

  auto start = std::chrono::steady_clock::now();
  for (int iter = 0; iter < repeat; iter++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int64_t i = 0; i < m * n; i++) d_dense[i] = 0.0f;

    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int64_t row = 0; row < m; row++) {
      for (int64_t idx = d_csr_off[row]; idx < d_csr_off[row+1]; idx++) {
        int64_t col = d_csr_col[idx];
        d_dense[row * n + col] = d_csr_val[idx];
      }
    }
  }
  auto end = std::chrono::steady_clock::now();
  printf("Average execution time of SparseToDense_convert : %f (us)\n",
         (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() * 1e-3f) / repeat);

  #pragma omp target update from(d_dense[0:m*n])
  for (int64_t i = 0; i < m*n; i++) h_dense_result[i] = d_dense[i];

  // Verify
  int correct = 1;
  int64_t nnz = 0;
  for (int64_t i = 0; i < m*n; i++) {
    if (h_dense_result[i] != 0) nnz++;
    if (h_dense[i] != h_dense_result[i]) { correct = 0; break; }
  }
  if (correct && nnz != h_nnz) correct = 0;
  printf("%s\n", correct ? "sparse2dense_csr_example test PASSED"
                          : "sparse2dense_csr_example test FAILED: wrong result");

  #pragma omp target exit data map(delete: d_dense[0:m*n], d_csr_off[0:m+1], \
                                           d_csr_col[0:h_nnz], d_csr_val[0:h_nnz])
  free(h_dense); free(h_dense_result); free(h_csr_values); free(h_csr_columns);
  free(h_csr_offsets); free(d_dense);
  return 0;
}
