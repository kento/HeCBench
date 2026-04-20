// spmv – OpenMP target port of spmv-kokkos
// Sparse matrix-vector multiplication (CSR and COO formats)

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <chrono>

#define REAL float

static void init_vector(REAL *v, size_t n) {
  srand48(1 << 12);
  for (size_t i = 0; i < n; i++) v[i] = (REAL)drand48();
}

static void init_matrix(REAL *matrix, size_t num_rows, size_t nnz) {
  size_t num_elems = num_rows * num_rows;
  size_t *perm = (size_t*)malloc(num_elems * sizeof(size_t));
  for (size_t i = 0; i < num_elems; i++) perm[i] = i;
  for (size_t i = num_elems; i > 0; i--) {
    size_t a = i-1, b = (size_t)(drand48() * i);
    if (a != b) { auto t = perm[a]; perm[a] = perm[b]; perm[b] = t; }
  }
  for (size_t i = 0; i < num_elems; i++) matrix[i] = 0;
  for (size_t i = 0; i < nnz; i++) matrix[perm[i]] = (REAL)drand48();
  free(perm);
}

static void init_csr(size_t *row_indices, REAL *values, size_t *col_indices,
                     REAL *matrix, size_t num_rows, size_t nnz) {
  row_indices[0] = 0;
  size_t *cnts = (size_t*)calloc(num_rows, sizeof(size_t));
  size_t tmp = 0;
  for (size_t i = 0; i < num_rows; i++)
    for (size_t j = 0; j < num_rows; j++)
      if (matrix[i*num_rows+j] != 0) {
        values[tmp] = matrix[i*num_rows+j];
        col_indices[tmp] = j;
        tmp++; cnts[i]++;
      }
  for (size_t i = 1; i <= num_rows; i++) row_indices[i] = row_indices[i-1] + cnts[i-1];
  free(cnts);
}

static void init_coo(size_t *row_indices, REAL *values, size_t *col_indices,
                     REAL *matrix, size_t num_rows, size_t nnz) {
  size_t tmp = 0;
  for (size_t i = 0; i < num_rows; i++)
    for (size_t j = 0; j < num_rows; j++)
      if (matrix[i*num_rows+j] != 0) {
        row_indices[tmp] = i;
        values[tmp] = matrix[i*num_rows+j];
        col_indices[tmp] = j;
        tmp++;
      }
}

static float check(REAL *A, REAL *B, size_t n) {
  float err = 0;
  for (size_t i = 0; i < n; i++) err += fabsf(A[i] - B[i]);
  return err / n;
}

static long reference(int repeat, size_t num_rows, const REAL *x, REAL *matrix, REAL *y) {
  REAL *d_x   = (REAL*)malloc(num_rows * sizeof(REAL));
  REAL *d_mat = (REAL*)malloc(num_rows * num_rows * sizeof(REAL));
  REAL *d_y   = (REAL*)malloc(num_rows * sizeof(REAL));
  for (size_t i = 0; i < num_rows; i++) d_x[i] = x[i];
  for (size_t i = 0; i < num_rows*num_rows; i++) d_mat[i] = matrix[i];

  #pragma omp target enter data map(to: d_x[0:num_rows], d_mat[0:num_rows*num_rows]) \
                                map(alloc: d_y[0:num_rows])

  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (size_t i = 0; i < num_rows; i++) {
      REAL temp = 0;
      for (size_t j = 0; j < num_rows; j++)
        if (d_mat[i*num_rows+j] != 0) temp += d_mat[i*num_rows+j] * d_x[j];
      d_y[i] = temp;
    }
  }
  auto end = std::chrono::steady_clock::now();

  #pragma omp target update from(d_y[0:num_rows])
  for (size_t i = 0; i < num_rows; i++) y[i] = d_y[i];

  #pragma omp target exit data map(delete: d_x[0:num_rows], d_mat[0:num_rows*num_rows], d_y[0:num_rows])
  free(d_x); free(d_mat); free(d_y);
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

static long spmv_csr(int repeat, size_t num_rows, const REAL *x, size_t nnz,
                     REAL *matrix, REAL *y) {
  size_t *row_idx=(size_t*)malloc((num_rows+1)*sizeof(size_t));
  size_t *col_idx=(size_t*)malloc(nnz*sizeof(size_t));
  REAL   *vals   =(REAL*)malloc(nnz*sizeof(REAL));
  init_csr(row_idx, vals, col_idx, matrix, num_rows, nnz);

  REAL   *d_x   = (REAL*)  malloc(num_rows * sizeof(REAL));
  REAL   *d_y   = (REAL*)  malloc(num_rows * sizeof(REAL));
  for (size_t i = 0; i < num_rows; i++) d_x[i] = x[i];

  #pragma omp target enter data \
    map(to: row_idx[0:num_rows+1], col_idx[0:nnz], vals[0:nnz], d_x[0:num_rows]) \
    map(alloc: d_y[0:num_rows])

  auto start = std::chrono::steady_clock::now();
  for (int r=0;r<repeat;r++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (size_t i = 0; i < num_rows; i++) {
      REAL temp = 0;
      for (size_t idx=row_idx[i]; idx<row_idx[i+1]; idx++) temp += vals[idx]*d_x[col_idx[idx]];
      d_y[i] = temp;
    }
  }
  auto end = std::chrono::steady_clock::now();

  #pragma omp target update from(d_y[0:num_rows])
  for (size_t i=0;i<num_rows;i++) y[i]=d_y[i];

  #pragma omp target exit data map(delete: row_idx[0:num_rows+1], col_idx[0:nnz], \
                                           vals[0:nnz], d_x[0:num_rows], d_y[0:num_rows])
  free(row_idx); free(col_idx); free(vals); free(d_x); free(d_y);
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
}

static long spmv_coo(int repeat, size_t num_rows, const REAL *x, size_t nnz,
                     REAL *matrix, REAL *y) {
  size_t *row_idx=(size_t*)malloc(nnz*sizeof(size_t));
  size_t *col_idx=(size_t*)malloc(nnz*sizeof(size_t));
  REAL   *vals   =(REAL*)malloc(nnz*sizeof(REAL));
  init_coo(row_idx, vals, col_idx, matrix, num_rows, nnz);

  REAL   *d_x  = (REAL*)malloc(num_rows*sizeof(REAL));
  REAL   *d_y  = (REAL*)calloc(num_rows, sizeof(REAL));
  for (size_t i = 0; i < num_rows; i++) d_x[i] = x[i];

  #pragma omp target enter data \
    map(to: row_idx[0:nnz], col_idx[0:nnz], vals[0:nnz], d_x[0:num_rows]) \
    map(alloc: d_y[0:num_rows])

  auto start = std::chrono::steady_clock::now();
  for (int r=0;r<repeat;r++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (size_t i = 0; i < num_rows; i++) d_y[i] = 0;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (size_t i = 0; i < nnz; i++) {
      #pragma omp atomic
      d_y[row_idx[i]] += vals[i]*d_x[col_idx[i]];
    }
  }
  auto end = std::chrono::steady_clock::now();

  #pragma omp target update from(d_y[0:num_rows])
  for (size_t i=0;i<num_rows;i++) y[i]=d_y[i];

  #pragma omp target exit data map(delete: row_idx[0:nnz], col_idx[0:nnz], vals[0:nnz], \
                                           d_x[0:num_rows], d_y[0:num_rows])
  free(row_idx); free(col_idx); free(vals); free(d_x); free(d_y);
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage %s <number of non-zero elements> <number of rows in a square matrix> <repeat>\n", argv[0]);
    return 1;
  }
  size_t nnz = atol(argv[1]);
  size_t num_rows = atol(argv[2]);
  int repeat = atoi(argv[3]);
  assert(nnz > 0 && num_rows > 0);
  assert(nnz <= num_rows * num_rows);

  REAL *x = (REAL*)malloc(num_rows * sizeof(REAL));
  REAL *y_ref = (REAL*)malloc(num_rows * sizeof(REAL));
  REAL *y = (REAL*)malloc(num_rows * sizeof(REAL));
  REAL *matrix = (REAL*)malloc(num_rows * num_rows * sizeof(REAL));

  srand48(1 << 12);
  init_matrix(matrix, num_rows, nnz);
  init_vector(x, num_rows);

  long elapsed = reference(repeat, num_rows, x, matrix, y_ref);

  printf("Number of non-zero elements: %zu\n", nnz);
  printf("Number of rows in a square matrix: %zu\n", num_rows);
  printf("Sparsity: %lf%%\n", (num_rows*num_rows - nnz) * 1.0 / (num_rows*num_rows) * 100.0);

  elapsed = spmv_csr(repeat, num_rows, x, nnz, matrix, y);
  printf("Average kernel (CSR) execution time (ms): %lf\n", elapsed * 1e-6 / repeat);
  printf("Error rate: %f\n", check(y, y_ref, num_rows));

  elapsed = spmv_coo(repeat, num_rows, x, nnz, matrix, y);
  printf("Average kernel (COO) execution time (ms): %lf\n", elapsed * 1e-6 / repeat);
  printf("Error rate: %f\n", check(y, y_ref, num_rows));

  free(x); free(y); free(y_ref); free(matrix);
  return 0;
}
