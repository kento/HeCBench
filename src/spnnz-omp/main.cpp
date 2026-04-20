// spnnz – OpenMP target port of spnnz-kokkos
// Count non-zero elements per row and total in a dense matrix

#include <omp.h>
#include <cstdio>
#include <cstdlib>
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

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("Counts non-zero elements per row and total in a dense matrix\n");
    printf("Usage %s <M> <N> <nnz> <repeat>\n", argv[0]);
    return 1;
  }
  int64_t m = atol(argv[1]);
  int64_t n = atol(argv[2]);
  int64_t h_nnz = atol(argv[3]);
  int repeat = atoi(argv[4]);

  float *h_dense = (float*)malloc(m * n * sizeof(float));
  int   *nnzPerRow = (int*)malloc(m * sizeof(int));

  printf("Initializing host matrices..\n");
  init_matrix(h_dense, m, n, h_nnz);

  float *d_dense  = h_dense;
  int   *d_nnzPR  = nnzPerRow;

  #pragma omp target enter data map(to: d_dense[0:m*n]) \
                                map(alloc: d_nnzPR[0:m])

  auto start = std::chrono::steady_clock::now();
  for (int iter = 0; iter < repeat; iter++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int64_t row = 0; row < m; row++) {
      int cnt = 0;
      for (int64_t j = 0; j < n; j++)
        if (d_dense[row * n + j] != 0.0f) cnt++;
      d_nnzPR[row] = cnt;
    }
  }
  auto end = std::chrono::steady_clock::now();
  printf("Average execution time of cusparseSnnz : %f (us)\n",
         (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() * 1e-3f) / repeat);

  // Count total nnz
  int nnzTotal = 0;
  #pragma omp target teams distribute parallel for reduction(+:nnzTotal) thread_limit(256)
  for (int64_t i = 0; i < m; i++) nnzTotal += d_nnzPR[i];

  #pragma omp target update from(d_nnzPR[0:m])

  // Verify
  int correct = 1;
  if (h_nnz != (int64_t)nnzTotal) { printf("nnz: %ld != %d\n", h_nnz, nnzTotal); correct = 0; }
  if (correct) {
    for (int64_t i = 0; i < m; i++) {
      int ref = 0;
      for (int64_t j = 0; j < n; j++) if (h_dense[i*n+j] != 0) ref++;
      if (ref != nnzPerRow[i]) { correct = 0; break; }
    }
  }
  printf("%s\n", correct ? "sparse_nnz_example test PASSED" : "sparse_nnz_example test FAILED: wrong result");

  #pragma omp target exit data map(delete: d_dense[0:m*n], d_nnzPR[0:m])
  free(h_dense); free(nnzPerRow);
  return 0;
}
