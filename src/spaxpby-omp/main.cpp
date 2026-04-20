// spaxpby – OpenMP target port of spaxpby-kokkos
// Sparse AXPBY: Y[i] = beta*Y[i] + alpha*X[sparse]

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

static void init_matrix(float *matrix, size_t num_rows, size_t num_cols, size_t nnz) {
  size_t n = num_rows * num_cols;
  float *d = (float*)malloc(n * sizeof(float));
  srand(123);
  for (size_t i = 0; i < n; i++) d[i] = (float)i;
  for (size_t i = n; i > 0; i--) {
    size_t a = i - 1, b = rand() % i;
    if (a != b) { auto t = d[a]; d[a] = d[b]; d[b] = t; }
  }
  srand48(123);
  for (size_t i = 0; i < num_rows; i++)
    for (size_t j = 0; j < num_cols; j++)
      matrix[i*num_cols+j] = (d[i*num_cols+j] >= (float)nnz) ? 0.0f : (float)(drand48()+1);
  free(d);
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("The function computes the sum of a sparse vector and a dense vector\n");
    printf("Y[i] = beta*Y[i] for all i, then Y[X_indices[k]] += alpha*X_values[k]\n");
    printf("Usage %s <M> <N> <nnz> <repeat>\n", argv[0]);
    return 1;
  }

  size_t m   = atol(argv[1]);
  size_t n   = atol(argv[2]);
  size_t nnz = atol(argv[3]);
  int repeat = atoi(argv[4]);

  const size_t size = m * n;
  float  *hA        = (float*)malloc(size * sizeof(float));
  float  *hB        = (float*)malloc(size * sizeof(float));
  float  *hY        = (float*)malloc(size * sizeof(float));
  float  *hA_values  = (float*)malloc(nnz * sizeof(float));
  size_t *hA_indices = (size_t*)malloc(nnz * sizeof(size_t));

  printf("Initializing input matrices..\n");
  init_matrix(hA, m, n, nnz);

  size_t k = 0;
  for (size_t i = 0; i < size; i++) {
    if (hA[i] != 0.0f) { hA_indices[k] = i; hA_values[k] = hA[i]; k++; }
  }
  init_matrix(hB, m, n, size);
  printf("Done\n");

  const float alpha = 1.0f, beta = 1.0f;

  float  *dX_values  = hA_values;
  size_t *dX_indices = hA_indices;
  float  *dY         = (float*)malloc(size * sizeof(float));
  for (size_t i = 0; i < size; i++) dY[i] = hB[i];

  #pragma omp target enter data map(to: dX_values[0:nnz], dX_indices[0:nnz], dY[0:size])

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (size_t j = 0; j < size; j++) dY[j] *= beta;

    #pragma omp target teams distribute parallel for thread_limit(256)
    for (size_t j = 0; j < nnz; j++) {
      #pragma omp atomic
      dY[dX_indices[j]] += alpha * dX_values[j];
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SPAXPBY : %f (us)\n", (time * 1e-3f) / repeat);

  #pragma omp target update from(dY[0:size])
  #pragma omp target exit data map(delete: dX_values[0:nnz], dX_indices[0:nnz], dY[0:size])
  for (size_t i = 0; i < size; i++) hY[i] = dY[i];
  free(dY);

  printf("Computing the reference results..\n");
  for (int iter = 0; iter < repeat; iter++) {
    for (size_t i = 0; i < size; i++)
      hB[i] = alpha * hA[i] + beta * hB[i];
  }
  printf("Done\n");

  int correct = 1;
  for (size_t i = 0; i < size; i++) {
    if (fabsf(hY[i] - hB[i]) > 1e-2f) { correct = 0; break; }
  }
  printf("%s\n", correct ? "axpby_example test PASSED" : "axpby_example test FAILED: wrong result");

  free(hA); free(hB); free(hY); free(hA_values); free(hA_indices);
  return 0;
}
