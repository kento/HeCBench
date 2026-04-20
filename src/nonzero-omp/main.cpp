#include <omp.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define MAX_DIMS 2

// Write multi-dimensional indices from flat indices stored in d_out[0:nzero].
// After the call d_out[0:nzero] holds row indices and d_out[nzero:2*nzero] holds column indices.
template <typename index_t>
void write_indices(int64_t *d_out, const index_t sizes[MAX_DIMS], int /*ndim*/, index_t nzero)
{
  index_t s0 = sizes[0], s1 = sizes[1];
#pragma omp target teams distribute parallel for thread_limit(256)
  for (int64_t idx = 0; idx < nzero; idx++) {
    int64_t idx_flat = d_out[idx];
    d_out[idx + 1 * nzero] = idx_flat % s1;
    d_out[idx + 0 * nzero] = (idx_flat / s1) % s0;
  }
}

template <typename scalar_t>
void nonzero(int nrows, int ncols, int repeat)
{
  const int in_ndims = MAX_DIMS;
  int64_t in_sizes[MAX_DIMS] = {nrows, ncols};
  int64_t num_items = (int64_t)nrows * ncols;

  std::mt19937 gen(19937);
  std::uniform_int_distribution<> dist(-1, 1);
  std::vector<scalar_t> h_in(num_items);

  bool ok = true;
  long sum_time = 0, idx_time = 0;

  for (int n = 0; n < repeat; n++) {
    int64_t r_nzeros = 0;
    do {
      r_nzeros = 0;
      for (int64_t i = 0; i < num_items; i++) {
        h_in[i] = static_cast<scalar_t>(dist(gen));
        if (h_in[i] != static_cast<scalar_t>(0)) r_nzeros++;
      }
    } while (r_nzeros == 0);

    scalar_t *d_in = (scalar_t *)malloc(num_items * sizeof(scalar_t));
    for (int64_t i = 0; i < num_items; i++) d_in[i] = h_in[i];
#pragma omp target enter data map(alloc: d_in[0:num_items])
#pragma omp target update to(d_in[0:num_items])

    int64_t h_nzeros = 0;
    auto start = std::chrono::steady_clock::now();
#pragma omp target teams distribute parallel for reduction(+:h_nzeros) thread_limit(256)
    for (int64_t i = 0; i < num_items; i++) {
      if (d_in[i] != static_cast<scalar_t>(0)) h_nzeros++;
    }
    auto end = std::chrono::steady_clock::now();
    sum_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    if (h_nzeros != r_nzeros) {
      printf("Number of non-zero elements mismatch: %ld != %ld (expected)\n",
             (long)h_nzeros, (long)r_nzeros);
      ok = false;
    } else {
      int64_t d_out_size = h_nzeros * in_ndims;
      int64_t *d_out = (int64_t *)malloc(d_out_size * sizeof(int64_t));
#pragma omp target enter data map(alloc: d_out[0:d_out_size])

      start = std::chrono::steady_clock::now();
      // Sequential prefix scan on host: collect flat indices of nonzero elements.
      // h_in is already available so no device download is needed.
      int64_t pos = 0;
      for (int64_t i = 0; i < num_items; i++) {
        if (h_in[i] != static_cast<scalar_t>(0))
          d_out[pos++] = i;
      }
      // Upload flat indices to device, then expand to multi-dimensional indices.
#pragma omp target update to(d_out[0:h_nzeros])
      write_indices<int64_t>(d_out, in_sizes, in_ndims, h_nzeros);
      end = std::chrono::steady_clock::now();
      idx_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

      // Verify: download full d_out (row indices + col indices) and check.
#pragma omp target update from(d_out[0:d_out_size])
      int64_t cnt_nzero = 0;
      for (int64_t i = 0; i < h_nzeros; i++) {
        int64_t row = d_out[i], col = d_out[h_nzeros + i];
        if (h_in[in_sizes[1] * row + col] != static_cast<scalar_t>(0)) cnt_nzero++;
      }
      ok = (cnt_nzero == h_nzeros);

#pragma omp target exit data map(delete: d_out[0:d_out_size])
      free(d_out);
    }

#pragma omp target exit data map(delete: d_in[0:num_items])
    free(d_in);

    if (!ok) break;
  }

  printf("Average time for sum reduction: %lf (us)\n", sum_time * 1e-3 / repeat);
  printf("Average time for write index operations: %lf (us)\n", idx_time * 1e-3 / repeat);
  printf("%s\n", ok ? "PASS" : "FAIL");
}

int main(int argc, char *argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of rows> <number of columns> <repeat>\n", argv[0]);
    return 1;
  }
  int nrows = atoi(argv[1]), ncols = atoi(argv[2]), repeat = atoi(argv[3]);
  if (nrows <= 0) nrows = 1;
  if (ncols <= 0) ncols = 1;

  for (int w = 0; w < 2; w++) {
    printf("=========== Data type is I8 ==========\n");
    nonzero<int8_t>(nrows, ncols, repeat);
    printf("=========== Data type is I16 ==========\n");
    nonzero<int16_t>(nrows, ncols, repeat);
    printf("=========== Data type is I32 ==========\n");
    nonzero<int32_t>(nrows, ncols, repeat);
    printf("=========== Data type is FP32 ==========\n");
    nonzero<float>(nrows, ncols, repeat);
    printf("=========== Data type is FP64 ==========\n");
    nonzero<double>(nrows, ncols, repeat);
  }
  return 0;
}
