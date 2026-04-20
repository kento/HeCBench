// spsort – OpenMP target port of spsort-cuda
// Sort column indices within each CSR row (sort-by-key: column index → value)

#include <omp.h>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <algorithm>

// Inline versions of the sycl/cuda utility functions
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
  row_indices[0] = 0; row_indices[num_rows] = nnz;
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

// Shuffle column indices within each row (to create unsorted input)
static void shuffle_matrix_data(const int *ia, int *ja, float *a, int nrows, int nnz) {
  for (int i = 0; i < nrows; i++) {
    int nnz_row = ia[i+1] - ia[i];
    for (int j = ia[i]; j < ia[i+1]; j++) {
      int q = ia[i] + std::rand() % nnz_row;
      std::swap(ja[q], ja[j]);
      std::swap(a[q],  a[j]);
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    std::cout << "Sorts column indices of each row in a sparse MxN matrix (CSR format)" << std::endl;
    std::cout << "Usage " << argv[0] << " <M> <N> <nnz> <repeat>" << std::endl;
    return 1;
  }
  int m = atoi(argv[1]), n = atoi(argv[2]), nnz = atoi(argv[3]), repeat = atoi(argv[4]);

  float *mat = (float*)malloc(sizeof(float) * m * n);
  init_matrix(mat, m, n, nnz);

  int   *ia = (int*)  malloc((m+1) * sizeof(int));
  int   *ja = (int*)  malloc(nnz   * sizeof(int));
  float *a  = (float*)malloc(nnz   * sizeof(float));
  init_csr(ia, a, ja, mat, m, n, nnz);

  int   *ia_r = (int*)  malloc((m+1) * sizeof(int));
  int   *ja_r = (int*)  malloc(nnz   * sizeof(int));
  float *a_r  = (float*)malloc(nnz   * sizeof(float));
  memcpy(ia_r, ia, (m+1)*sizeof(int));
  memcpy(ja_r, ja, nnz  *sizeof(int));
  memcpy(a_r,  a,  nnz  *sizeof(float));
  shuffle_matrix_data(ia_r, ja_r, a_r, m, nnz);

  // Device sort: for each row sort (ja_r, a_r) by ja_r using insertion sort on device
  int   *d_ia = ia_r;
  int   *d_ja = ja_r;
  float *d_a  = a_r;

  #pragma omp target enter data \
    map(to: d_ia[0:m+1], d_ja[0:nnz], d_a[0:nnz])

  auto start = std::chrono::steady_clock::now();
  for (int rep = 0; rep < repeat; rep++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int row = 0; row < m; row++) {
      int beg = d_ia[row], end2 = d_ia[row+1];
      // Insertion sort within the row
      for (int i = beg + 1; i < end2; i++) {
        int   key_idx = d_ja[i];
        float key_val = d_a[i];
        int j = i - 1;
        while (j >= beg && d_ja[j] > key_idx) {
          d_ja[j+1] = d_ja[j];
          d_a[j+1]  = d_a[j];
          j--;
        }
        d_ja[j+1] = key_idx;
        d_a[j+1]  = key_val;
      }
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
  std::cout << "Average execution time of CSR sort: " << (time * 1e-3f) / repeat << " us" << std::endl;

  #pragma omp target update from(d_ia[0:m+1], d_ja[0:nnz], d_a[0:nnz])

  // Verify
  bool error = false;
  if (memcmp(ia_r, ia, (m+1)*sizeof(int)) != 0) { std::cout << "Error: row index arrays mismatch" << std::endl; error=true; }
  if (!error && memcmp(ja_r, ja, nnz*sizeof(int))   != 0) { std::cout << "Error: column index arrays mismatch" << std::endl; error=true; }
  if (!error && memcmp(a_r,  a,  nnz*sizeof(float)) != 0) { std::cout << "Error: value arrays mismatch" << std::endl; error=true; }

  if (!error) std::cout << "csrsort_example test PASSED" << std::endl;
  else        std::cout << "csrsort_example test FAILED: wrong result" << std::endl;

  #pragma omp target exit data map(delete: d_ia[0:m+1], d_ja[0:nnz], d_a[0:nnz])
  free(mat); free(ia); free(ja); free(a); free(ia_r); free(ja_r); free(a_r);
  return 0;
}
