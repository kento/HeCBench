// spd2s – OpenMP target port of spd2s-kokkos
// Dense-to-sparse (CSR) conversion

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <vector>

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
  int64_t tmp = 0;
  for (int64_t i = 0; i < num_rows; i++) {
    int64_t cnt = 0;
    for (int64_t j = 0; j < num_cols; j++) {
      if (matrix[i*num_cols+j] != 0.0f) {
        values[tmp] = matrix[i*num_cols+j];
        col_indices[tmp] = j;
        tmp++; cnt++;
      }
    }
    row_offsets[i+1] = row_offsets[i] + cnt;
  }
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("The function converts a dense MxN matrix into a sparse CSR matrix\n");
    printf("Usage %s <M> <N> <nnz> <repeat>\n", argv[0]);
    return 1;
  }

  int64_t m = atol(argv[1]);
  int64_t n = atol(argv[2]);
  int64_t h_nnz = atol(argv[3]);
  int repeat = atoi(argv[4]);

  float   *h_dense        = (float*)  malloc(m * n * sizeof(float));
  float   *h_csr_values   = (float*)  malloc(h_nnz * sizeof(float));
  int64_t *h_csr_columns  = (int64_t*)malloc(h_nnz * sizeof(int64_t));
  int64_t *h_csr_offsets  = (int64_t*)malloc((m+1) * sizeof(int64_t));
  float   *h_csr_values_ref  = (float*)  malloc(h_nnz * sizeof(float));
  int64_t *h_csr_columns_ref = (int64_t*)malloc(h_nnz * sizeof(int64_t));
  int64_t *h_csr_offsets_ref = (int64_t*)malloc((m+1) * sizeof(int64_t));

  printf("Initializing host matrices..\n");
  init_matrix(h_dense, m, n, h_nnz);
  init_csr(h_csr_offsets_ref, h_csr_values_ref, h_csr_columns_ref, h_dense, m, n, h_nnz);

  float   *d_dense       = h_dense;
  int64_t *d_csr_offsets = (int64_t*)malloc((m+1) * sizeof(int64_t));
  int64_t *d_csr_columns = (int64_t*)malloc(h_nnz * sizeof(int64_t));
  float   *d_csr_values  = (float*)  malloc(h_nnz * sizeof(float));

  #pragma omp target enter data map(to: d_dense[0:m*n]) \
    map(alloc: d_csr_offsets[0:m+1], d_csr_columns[0:h_nnz], d_csr_values[0:h_nnz])

  auto start = std::chrono::steady_clock::now();

  for (int iter = 0; iter < repeat; iter++) {
    // Count nnz per row
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int64_t row = 0; row < m; row++) {
      int64_t cnt = 0;
      for (int64_t j = 0; j < n; j++)
        if (d_dense[row * n + j] != 0.0f) cnt++;
      d_csr_offsets[row + 1] = cnt;
    }

    // Prefix scan on host
    #pragma omp target update from(d_csr_offsets[0:m+1])
    d_csr_offsets[0] = 0;
    for (int64_t i = 0; i < m; i++) d_csr_offsets[i+1] += d_csr_offsets[i];
    #pragma omp target update to(d_csr_offsets[0:m+1])

    // Fill CSR
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int64_t row = 0; row < m; row++) {
      int64_t pos = d_csr_offsets[row];
      for (int64_t j = 0; j < n; j++) {
        float v = d_dense[row * n + j];
        if (v != 0.0f) {
          d_csr_columns[pos] = j;
          d_csr_values[pos]  = v;
          pos++;
        }
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of DenseToSparse_convert : %f (us)\n", (time * 1e-3f) / repeat);

  #pragma omp target update from(d_csr_offsets[0:m+1], d_csr_columns[0:h_nnz], d_csr_values[0:h_nnz])
  #pragma omp target exit data map(delete: d_dense[0:m*n], d_csr_offsets[0:m+1], \
                                           d_csr_columns[0:h_nnz], d_csr_values[0:h_nnz])

  for (int64_t i = 0; i <= m; i++) h_csr_offsets[i]  = d_csr_offsets[i];
  for (int64_t i = 0; i < h_nnz; i++) {
    h_csr_columns[i] = d_csr_columns[i];
    h_csr_values[i]  = d_csr_values[i];
  }

  int correct = 1;
  int64_t nnz = h_csr_offsets[m];
  if (nnz != h_nnz) { printf("nnz: %ld != %ld\n", nnz, h_nnz); correct = 0; }

  if (correct) {
    std::vector<int64_t> cols(h_csr_columns, h_csr_columns + h_nnz);
    std::vector<float>   vals(h_csr_values,  h_csr_values  + h_nnz);
    std::vector<int64_t> cols_ref(h_csr_columns_ref, h_csr_columns_ref + h_nnz);
    std::vector<float>   vals_ref(h_csr_values_ref,  h_csr_values_ref  + h_nnz);
    std::sort(cols.begin(), cols.end());
    std::sort(vals.begin(), vals.end());
    std::sort(cols_ref.begin(), cols_ref.end());
    std::sort(vals_ref.begin(), vals_ref.end());
    for (int64_t i = 0; i < h_nnz; i++) {
      if (cols[i] != cols_ref[i] || fabsf(vals[i] - vals_ref[i]) > 1e-5f) {
        correct = 0; break;
      }
    }
  }
  printf("%s\n", correct ? "dense2sparse_csr_example test PASSED"
                          : "dense2sparse_csr_example test FAILED: wrong result");

  free(h_dense); free(h_csr_values); free(h_csr_columns); free(h_csr_offsets);
  free(h_csr_values_ref); free(h_csr_columns_ref); free(h_csr_offsets_ref);
  free(d_csr_offsets); free(d_csr_columns); free(d_csr_values);
  return 0;
}
