// sparkler – OpenMP target port of sparkler-kokkos
// Simplified single-process GEMM benchmark: C = A * B^T

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>

#pragma omp declare target
static size_t nonzero_stride(const size_t i) {
  enum { MAX = 499 };
  return 1 + i % MAX;
}
#pragma omp end declare target

void set_input_matrix(float *mat, size_t nr, size_t nc,
                      size_t nru, size_t base_vector_num, float value) {
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (size_t index = 0; index < nr * nc; index++) {
    const size_t r = index % nr;
    const size_t c = index / nr;
    const size_t stride = nonzero_stride(r + base_vector_num);
    mat[r + nru * c] = (c % stride == 0) ? value : 0.0f;
  }
}

// GEMM: C = A * B^T  (m x n = m x k * k x n^T)
void perform_gemm(float *A, float *B, float *C,
                  size_t m, size_t n, size_t k,
                  size_t nru_a, size_t nru_b, size_t nru_c) {
  #pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
  for (size_t r = 0; r < m; r++) {
    for (size_t c = 0; c < n; c++) {
      float sum = 0.0f;
      for (size_t kk = 0; kk < k; kk++)
        sum += A[r + nru_a * kk] * B[c + nru_b * kk];
      C[r + nru_c * c] = sum;
    }
  }
}

static size_t roundup8(size_t x) { return ((x + 7) / 8) * 8; }

int main(int argc, char** argv) {
  size_t num_vector = 0;
  size_t num_field = 0;
  int num_iterations = 1;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--num_vector") == 0 && i+1 < argc) num_vector = atol(argv[++i]);
    if (strcmp(argv[i], "--num_field") == 0 && i+1 < argc)  num_field  = atol(argv[++i]);
    if (strcmp(argv[i], "--num_iterations") == 0 && i+1 < argc) num_iterations = atoi(argv[++i]);
  }

  if (num_vector < 2 || num_field < 1) {
    printf("Usage: %s --num_vector <n> --num_field <n> [--num_iterations <n>]\n", argv[0]);
    printf("  num_vector >= 2, num_field >= 1\n");
    return 1;
  }

  printf("num_vector %zu num_field %zu num_iterations %d num_proc 1\n",
         num_vector, num_field, num_iterations);

  const size_t m   = 2 * roundup8(num_vector);
  const size_t k   = num_field;
  const size_t n   = m;
  const size_t nru = roundup8(m);

  size_t szA = roundup8(m) * roundup8(k);
  size_t szB = roundup8(n) * roundup8(k);
  size_t szC = roundup8(m) * roundup8(n);

  float *A = (float*)malloc(szA * sizeof(float));
  float *B = (float*)malloc(szB * sizeof(float));
  float *C = (float*)malloc(szC * sizeof(float));

  memset(C, 0, szC * sizeof(float));

  #pragma omp target enter data map(alloc: A[0:szA], B[0:szB], C[0:szC])

  set_input_matrix(A, m, k, roundup8(m), 0, 1.0f);
  set_input_matrix(B, n, k, roundup8(n), 0, 1.0f);

  double timegemm = 0.0;
  double flops = 0.0;

  auto tstart = std::chrono::steady_clock::now();

  for (int iteration = 1; iteration <= num_iterations; ++iteration) {
    auto t1 = std::chrono::steady_clock::now();

    perform_gemm(A, B, C, m, n, k, roundup8(m), roundup8(n), roundup8(m));
    flops += 2.0 * m * n * k;

    auto t2 = std::chrono::steady_clock::now();
    timegemm += std::chrono::duration<double>(t2 - t1).count();

    if (!(iteration & (iteration-1)) || iteration % 256 == 0 || iteration == num_iterations) {
      double elapsed = std::chrono::duration<double>(t2 - tstart).count();
      printf("Iteration %d of %d, elapsed sec %.3f\n", iteration, num_iterations, elapsed);
    }
  }

  auto tend = std::chrono::steady_clock::now();
  double timetotal = std::chrono::duration<double>(tend - tstart).count();

  printf("TFLOPS %.3f\nGEMM time %.3f (s)\nGEMM TFLOPS/s %.3f\nTotal time %.3f (s)\n",
         flops/1e12, timegemm, flops*1e-12/timegemm, timetotal);

  #pragma omp target exit data map(delete: A[0:szA], B[0:szB], C[0:szC])
  free(A); free(B); free(C);
  return 0;
}
