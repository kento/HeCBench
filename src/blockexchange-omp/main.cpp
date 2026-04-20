#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>

// Perform a blocked-to-striped rearrangement over n elements.
//
// Within each block of `items_per_block = block_size * items_per_thread`
// consecutive elements starting at position k:
//   Input  (blocked):  thread t owns positions k + t*IPT .. k + t*IPT + (IPT-1)
//   Output (striped):  thread t owns positions k + t, k + BLK+t, k + 2*BLK+t, ...
//
// Reading the output sequentially: out[k + p] = in[k + (p%IPT)*block_size + p/IPT]
//
// Equivalently for each output index idx:
//   k   = block-start
//   p   = idx - k
//   t   = p / IPT   (thread that "owned" this slot in blocked layout)
//   j   = p % IPT   (item index within that thread)
//   src = k + j*block_size + t
//   out[idx] = in[src]  (if src < n, else 0)

void blocked_to_striped(const int* d_in,
                        int*       d_out,
                        const int n,
                        const int block_size,
                        const int items_per_thread)
{
  const int items_per_block = block_size * items_per_thread;

  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int idx = 0; idx < n; idx++) {
    const int k   = (idx / items_per_block) * items_per_block;
    const int p   = idx - k;
    const int t   = p / items_per_thread;
    const int j   = p % items_per_thread;
    const int src = k + j * block_size + t;
    d_out[idx] = (src < n) ? d_in[src] : 0;
  }
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of rows> <number of columns> <repeat>\n", argv[0]);
    return 1;
  }
  const int nrows  = atoi(argv[1]);
  const int ncols  = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const int n = nrows * ncols;
  constexpr int block_size        = 256;
  constexpr int items_per_thread  = 4;
  const int     items_per_block   = block_size * items_per_thread;

  int* d_A   = (int*)malloc(n * sizeof(int));
  int* d_out = (int*)malloc(n * sizeof(int));

  #pragma omp target enter data map(alloc: d_A[0:n], d_out[0:n])

  // Fill input on device: A[i] = i
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < n; i++) d_A[i] = i;

  // Benchmark
  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++)
    blocked_to_striped(d_A, d_out, n, block_size, items_per_thread);
  auto end = std::chrono::steady_clock::now();
  long long time_ns =
    std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kernel: %f (us)\n", (time_ns * 1e-3) / repeat);

  // Copy back and verify
  int* h_out = (int*)malloc(n * sizeof(int));
  #pragma omp target update from(d_out[0:n])
  memcpy(h_out, d_out, n * sizeof(int));

  if (n % items_per_block == 0) {
    bool ok = true;
  outer:
    for (int k = 0; k < n; k += items_per_block) {
      int i = k;
      for (int j = 0; j < items_per_thread; j++, i++) {
        for (int m = 0; m < block_size - 1; m++) {
          if (i + (m + 1) * items_per_thread < n) {
            int diff = h_out[i + (m + 1) * items_per_thread] -
                       h_out[i + m * items_per_thread];
            if (diff != 1) {
              printf("Error at index %d (diff=%d)\n",
                     i + (m + 1) * items_per_thread, diff);
              ok = false;
              goto outer;
            }
          }
        }
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
  } else {
    printf("n is not a multiple of items_per_block; skipping verification.\n");
  }

  #pragma omp target exit data map(delete: d_A[0:n], d_out[0:n])
  free(d_A);
  free(d_out);
  free(h_out);
  return 0;
}
