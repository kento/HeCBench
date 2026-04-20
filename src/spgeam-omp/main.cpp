// spgeam – OpenMP target port of spgeam-kokkos
// CSR to CSC conversion (sparse matrix transpose)

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <vector>

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

static void transpose_dense(float *A, float *B, int rows, int cols) {
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      B[j*rows + i] = A[i*cols + j];
}

void csr2csc(int *d_csr_offsets, int *d_csr_columns, float *d_csr_values,
             int *d_csc_offsets, int *d_csc_rows, float *d_csc_values,
             int num_rows, int num_cols, int nnz)
{
  // Count nnz per column using atomics
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i <= num_cols; i++) d_csc_offsets[i] = 0;

  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < nnz; i++) {
    #pragma omp atomic
    d_csc_offsets[d_csr_columns[i] + 1]++;
  }

  // Prefix scan on host
  #pragma omp target update from(d_csc_offsets[0:num_cols+1])
  for (int i = 1; i <= num_cols; i++) d_csc_offsets[i] += d_csc_offsets[i-1];
  #pragma omp target update to(d_csc_offsets[0:num_cols+1])

  // Fill CSC rows and values (serial for correctness)
  #pragma omp target update from(d_csr_offsets[0:num_rows+1], d_csr_columns[0:nnz], d_csr_values[0:nnz])
  std::vector<int> work(d_csc_offsets, d_csc_offsets + num_cols + 1);
  std::vector<int> h_csc_row(nnz);
  std::vector<float> h_csc_val(nnz);
  for (int r = 0; r < num_rows; r++) {
    for (int idx = d_csr_offsets[r]; idx < d_csr_offsets[r+1]; idx++) {
      int c = d_csr_columns[idx];
      int pos = work[c]++;
      h_csc_row[pos] = r;
      h_csc_val[pos] = d_csr_values[idx];
    }
  }
  for (int i = 0; i < nnz; i++) { d_csc_rows[i] = h_csc_row[i]; d_csc_values[i] = h_csc_val[i]; }
  #pragma omp target update to(d_csc_rows[0:nnz], d_csc_values[0:nnz])
}

int main(int argc, char *argv[]) {
  if (argc != 6) {
    printf("CSR to CSC conversion (sparse matrix transpose)\n");
    printf("Usage %s <M> <K> <nnz> <repeat> <verify>\n", argv[0]);
    return 1;
  }

  int m = atoi(argv[1]);
  int k = atoi(argv[2]);
  int a_nnz = atoi(argv[3]);
  int repeat = atoi(argv[4]);
  int verify = atoi(argv[5]);

  float *hA = (float*)malloc(m * k * sizeof(float));
  float *hA_values = (float*)malloc(a_nnz * sizeof(float));
  int *hA_columns = (int*)malloc(a_nnz * sizeof(int));
  int *hA_offsets = (int*)malloc((m + 1) * sizeof(int));

  init_matrix(hA, m, k, a_nnz);
  init_csr(hA_offsets, hA_values, hA_columns, hA, m, k, a_nnz);

  int   *d_csr_offsets = hA_offsets;
  int   *d_csr_columns = hA_columns;
  float *d_csr_values  = hA_values;
  int   *d_csc_offsets = (int*)  malloc((k+1) * sizeof(int));
  int   *d_csc_rows    = (int*)  malloc(a_nnz * sizeof(int));
  float *d_csc_values  = (float*)malloc(a_nnz * sizeof(float));

  #pragma omp target enter data \
    map(to: d_csr_offsets[0:m+1], d_csr_columns[0:a_nnz], d_csr_values[0:a_nnz]) \
    map(alloc: d_csc_offsets[0:k+1], d_csc_rows[0:a_nnz], d_csc_values[0:a_nnz])

  auto start_t = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++)
    csr2csc(d_csr_offsets, d_csr_columns, d_csr_values,
            d_csc_offsets, d_csc_rows, d_csc_values,
            m, k, a_nnz);
  auto end_t = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_t - start_t).count();
  printf("Average execution time of cuSparse csr2cscEx2 : %f (us)\n", (time * 1e-3f) / repeat);

  if (verify) {
    printf("Computing the reference results..\n");
    float *hB = (float*)malloc(k * m * sizeof(float));
    transpose_dense(hA, hB, m, k);
    float *hB_values = (float*)malloc(a_nnz * sizeof(float));
    int *hB_rows = (int*)malloc(a_nnz * sizeof(int));
    int *hB_offsets = (int*)malloc((k+1) * sizeof(int));
    init_csr(hB_offsets, hB_values, hB_rows, hB, k, m, a_nnz);

    #pragma omp target update from(d_csc_offsets[0:k+1], d_csc_rows[0:a_nnz], d_csc_values[0:a_nnz])

    int correct = 1;
    for (int i = 0; i <= k; i++)
      if (d_csc_offsets[i] != hB_offsets[i]) { correct = 0; break; }
    if (correct) {
      for (int i = 0; i < a_nnz; i++)
        if (d_csc_rows[i] != hB_rows[i] || fabsf(d_csc_values[i] - hB_values[i]) > 1e-2f)
          { correct = 0; break; }
    }
    printf("%s\n", correct ? "spgeam_example test PASSED" : "spgeam_example test FAILED: wrong result");
    free(hB); free(hB_values); free(hB_rows); free(hB_offsets);
  }

  #pragma omp target exit data map(delete: d_csr_offsets[0:m+1], d_csr_columns[0:a_nnz], \
                                           d_csr_values[0:a_nnz], d_csc_offsets[0:k+1], \
                                           d_csc_rows[0:a_nnz], d_csc_values[0:a_nnz])
  free(hA); free(hA_values); free(hA_columns); free(hA_offsets);
  free(d_csc_offsets); free(d_csc_rows); free(d_csc_values);
  return 0;
}
