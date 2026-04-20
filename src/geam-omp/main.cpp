// OpenMP target offloading port of geam-kokkos (matrix transpose)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <omp.h>

template <typename T>
static void run_transpose(int nrow, int ncol, int repeat)
{
  const size_t size = (size_t)nrow * ncol;

  T* matrix  = (T*)malloc(size * sizeof(T));
  T* matrixT = (T*)malloc(size * sizeof(T));
  T* d_matrix  = (T*)malloc(size * sizeof(T));
  T* d_matrixT = (T*)malloc(size * sizeof(T));

  for (size_t i = 0; i < size; i++) matrix[i] = (T)(rand() % 13);

  // CPU reference
  for (int i = 0; i < nrow; i++)
    for (int j = 0; j < ncol; j++)
      matrixT[(size_t)j * nrow + i] = matrix[(size_t)i * ncol + j];

  for (size_t i = 0; i < size; i++) d_matrix[i] = matrix[i];

  #pragma omp target enter data map(to: d_matrix[0:size]) map(alloc: d_matrixT[0:size])

  const int warmup = 4;
  double time_ns = 0.0;

  for (int i = 0; i < warmup + repeat; i++) {
    auto start = std::chrono::steady_clock::now();

    #pragma omp target teams distribute parallel for thread_limit(256)
    for (size_t idx = 0; idx < size; idx++) {
      int r = (int)(idx / ncol);
      int c = (int)(idx % ncol);
      d_matrixT[(size_t)c * nrow + r] = d_matrix[idx];
    }

    auto end = std::chrono::steady_clock::now();
    if (i >= warmup)
      time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  printf("Device: average matrix transpose time = %f (ms)\n",
         time_ns * 1e-6 / repeat);

  #pragma omp target update from(d_matrixT[0:size])

  int error = memcmp(d_matrixT, matrixT, size * sizeof(T));
  printf("%s\n", error ? "FAIL" : "PASS");

  #pragma omp target exit data map(delete: d_matrix[0:size], d_matrixT[0:size])

  free(matrix); free(matrixT); free(d_matrix); free(d_matrixT);
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage %s <matrix row> <matrix col> <repeat>\n", argv[0]);
    return 1;
  }

  int nrow   = atoi(argv[1]);
  int ncol   = atoi(argv[2]);
  int repeat = atoi(argv[3]);

  if (nrow <= 0 || ncol <= 0 || repeat < 0) {
    printf("Error: invalid inputs\n"); return 1;
  }

  printf("---- FP32 transpose (%d x %d) ----\n", nrow, ncol);
  run_transpose<float>(nrow, ncol, repeat);

  printf("---- FP64 transpose (%d x %d) ----\n", nrow, ncol);
  run_transpose<double>(nrow, ncol, repeat);

  return 0;
}
