// OpenMP target offloading port of reverse2D benchmark.
// Reverses a 2D matrix along rows or columns for row-major and column-major layouts.

#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

// ---------------------------------------------------------------------------
// Reverse kernels – templated on element size via byte-level operations.
// We use unsigned long (8 bytes) as the widest type needed.
// ---------------------------------------------------------------------------

template <typename T>
static long reverseOMP(T* d_out, const T* d_in,
                        int nrows, int ncols,
                        bool rowMajor, bool alongRows,
                        int N_total)
{
  const int half_rows = (nrows + 1) / 2;
  const int half_cols = (ncols + 1) / 2;
  const int len       = alongRows ? half_rows * ncols : nrows * half_cols;

  auto t0 = std::chrono::steady_clock::now();

  if (rowMajor && alongRows) {
#pragma omp target teams distribute parallel for thread_limit(256)
    for (int idx = 0; idx < len; idx++) {
      int srcRow = idx / ncols;
      int srcCol = idx % ncols;
      int dstRow = nrows - srcRow - 1;
      int srcIdx = srcRow * ncols + srcCol;
      int dstIdx = dstRow * ncols + srcCol;
      T a = d_in[srcIdx];
      T b = d_in[dstIdx];
      d_out[dstIdx] = a;
      d_out[srcIdx] = b;
    }
  } else if (rowMajor && !alongRows) {
#pragma omp target teams distribute parallel for thread_limit(256)
    for (int idx = 0; idx < len; idx++) {
      int srcRow = idx / half_cols;
      int srcCol = idx % half_cols;
      int dstCol = ncols - srcCol - 1;
      int srcIdx = srcRow * ncols + srcCol;
      int dstIdx = srcRow * ncols + dstCol;
      T a = d_in[srcIdx];
      T b = d_in[dstIdx];
      d_out[dstIdx] = a;
      d_out[srcIdx] = b;
    }
  } else if (!rowMajor && alongRows) {
#pragma omp target teams distribute parallel for thread_limit(256)
    for (int idx = 0; idx < len; idx++) {
      int srcRow = idx % half_rows;
      int srcCol = idx / half_rows;
      int dstRow = nrows - srcRow - 1;
      int srcIdx = srcCol * nrows + srcRow;
      int dstIdx = srcCol * nrows + dstRow;
      T a = d_in[srcIdx];
      T b = d_in[dstIdx];
      d_out[dstIdx] = a;
      d_out[srcIdx] = b;
    }
  } else {
#pragma omp target teams distribute parallel for thread_limit(256)
    for (int idx = 0; idx < len; idx++) {
      int srcRow = idx % nrows;
      int srcCol = idx / nrows;
      int dstCol = ncols - srcCol - 1;
      int srcIdx = srcCol * nrows + srcRow;
      int dstIdx = dstCol * nrows + srcRow;
      T a = d_in[srcIdx];
      T b = d_in[dstIdx];
      d_out[dstIdx] = a;
      d_out[srcIdx] = b;
    }
  }

  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
}

template <typename T>
static void eval(int nrows, int ncols, int repeat)
{
  const int matrix_size = nrows * ncols;

  T* h_in  = new T[matrix_size];
  T* d_in  = new T[matrix_size];
  T* d_out = new T[matrix_size];

  std::default_random_engine rng(123);
  std::uniform_int_distribution<int> dist(0, 255);
  for (int i = 0; i < matrix_size; i++)
    h_in[i] = static_cast<T>(dist(rng));

  for (int i = 0; i < matrix_size; i++) d_in[i] = h_in[i];

#pragma omp target enter data map(to: d_in[0:matrix_size]) map(alloc: d_out[0:matrix_size])

  // 4 cases
  bool cases[4][2] = {{true,true},{true,false},{false,true},{false,false}};
  for (int c = 0; c < 4; c++) {
    bool rowMajor = cases[c][0], alongRows = cases[c][1];
    printf("\nInput matrix is %s-major, reverse along %s\n",
           rowMajor ? "row" : "column", alongRows ? "rows" : "columns");

    for (int i = 0; i < matrix_size; i++) d_in[i] = h_in[i];
#pragma omp target update to(d_in[0:matrix_size])

    long total_ns = 0;
    for (int i = 0; i < repeat; i++)
      total_ns += reverseOMP<T>(d_out, d_in, nrows, ncols, rowMajor, alongRows, matrix_size);

    printf("Average kernel execution time: %f (ms)\n", (double)total_ns * 1e-6 / repeat);
  }

#pragma omp target exit data map(delete: d_in[0:matrix_size], d_out[0:matrix_size])
  delete[] h_in; delete[] d_in; delete[] d_out;
}

int main(int argc, char** argv)
{
  if (argc != 4) {
    printf("Usage: %s <nrows> <ncols> <iterations>\n", argv[0]);
    return 1;
  }
  const int nrows  = atoi(argv[1]);
  const int ncols  = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  printf("\nElement size: %zu byte\n", sizeof(unsigned char));
  eval<unsigned char>(nrows, ncols, repeat);

  printf("\nElement size: %zu bytes\n", sizeof(unsigned short));
  eval<unsigned short>(nrows, ncols, repeat);

  printf("\nElement size: %zu bytes\n", sizeof(unsigned int));
  eval<unsigned int>(nrows, ncols, repeat);

  printf("\nElement size: %zu bytes\n", sizeof(unsigned long));
  eval<unsigned long>(nrows, ncols, repeat);

  return 0;
}
