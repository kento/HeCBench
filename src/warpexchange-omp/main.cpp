// OpenMP target offloading port of warpexchange benchmark
// Blocked-to-striped data rearrangement

#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void blocked_to_striped(const int *in, int *out, const int n,
                                const int block_size, const int items_per_thread) {
  const int tile = block_size * items_per_thread;
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int idx = 0; idx < n; idx++) {
    int tile_id   = idx / tile;
    int local_idx = idx % tile;
    int thread_id = local_idx / items_per_thread;
    int item_id   = local_idx % items_per_thread;
    int striped_local = item_id * block_size + thread_id;
    out[tile_id * tile + striped_local] = in[idx];
  }
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <nrows> <ncols> <repeat>\n", argv[0]);
    return 1;
  }
  const int nrows  = atoi(argv[1]);
  const int ncols  = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const int n = nrows * ncols;
  std::vector<int> h_in(n), h_out(n);
  for (int i = 0; i < n; i++) h_in[i] = i;

  int *d_in  = h_in.data();
  int *d_out = h_out.data();

  #pragma omp target enter data map(to: d_in[0:n]) map(alloc: d_out[0:n])

  const int block_size       = 256;
  const int items_per_thread = 4;

  // Warmup
  for (int i = 0; i < repeat; i++)
    blocked_to_striped(d_in, d_out, n, block_size, items_per_thread);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++)
    blocked_to_striped(d_in, d_out, n, block_size, items_per_thread);
  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time: %f (us)\n", time * 1e-3f / repeat);

  // Verify: compute reference
  std::vector<int> h_ref(n);
  int *d_ref = h_ref.data();
  #pragma omp target enter data map(alloc: d_ref[0:n])
  blocked_to_striped(d_in, d_ref, n, block_size, items_per_thread);
  #pragma omp target update from(d_ref[0:n], d_out[0:n])
  #pragma omp target exit data map(delete: d_in[0:n], d_out[0:n], d_ref[0:n])

  bool ok = true;
  for (int i = 0; i < n && ok; i++)
    if (h_ref[i] != h_out[i]) ok = false;
  printf("%s\n", ok ? "PASS" : "FAIL");
  return 0;
}
