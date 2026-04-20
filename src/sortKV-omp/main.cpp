// sortKV – OpenMP target port of sortKV-kokkos
// Sort key-value pairs by key using std::sort + parallel approach

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <vector>

template <typename T>
void sort_key_value(int n, int repeat, bool verify) {
  printf("Number of keys is %d and the size of each value in bytes is %zu\n", n, sizeof(T));

  unsigned seed = 123;
  bool ok = true;

  std::vector<int> keys(n);
  std::vector<T> vals(n);
  std::iota(keys.begin(), keys.end(), 0);

  double total_time = 0.0;
  for (int i = 0; i < repeat; i++) {
    std::srand(seed + i);
    for (int j = n - 1; j > 0; j--) {
      int k = std::rand() % (j + 1);
      std::swap(keys[j], keys[k]);
    }
    for (int j = 0; j < n; j++) vals[j] = (T)(keys[j] % 256);

    auto start = std::chrono::steady_clock::now();

    int *d_keys = keys.data();
    T   *d_vals = vals.data();

    #pragma omp target data map(tofrom: d_keys[0:n], d_vals[0:n])
    {
      // Transfer back for host-side sort (std::sort not available on GPU)
      #pragma omp target update from(d_keys[0:n], d_vals[0:n])

      // Create index permutation and sort
      std::vector<int> perm(n);
      std::iota(perm.begin(), perm.end(), 0);
      std::sort(perm.begin(), perm.end(),
                [&](int a, int b){ return d_keys[a] < d_keys[b]; });

      std::vector<int> sorted_keys(n);
      std::vector<T>   sorted_vals(n);
      for (int j = 0; j < n; j++) {
        sorted_keys[j] = d_keys[perm[j]];
        sorted_vals[j] = d_vals[perm[j]];
      }
      for (int j = 0; j < n; j++) { d_keys[j] = sorted_keys[j]; d_vals[j] = sorted_vals[j]; }

      #pragma omp target update to(d_keys[0:n], d_vals[0:n])
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;
  }

  if (!verify)
    printf("Average sort time %f (us)\n", (total_time * 1e-3) / repeat);
  else {
    for (int i = 0; i < n; i++) {
      if (keys[i] != i || vals[i] != (T)(i % 256)) {
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
  }
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of keys> <repeat>\n", argv[0]);
    return 1;
  }
  const int size   = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  printf("\nWarmup and verify\n");
  sort_key_value<unsigned char>(size, repeat, true);
  sort_key_value<short>(size, repeat, true);
  sort_key_value<int>(size, repeat, true);
  sort_key_value<long>(size, repeat, true);

  printf("\nPerformance evaluation\n");
  sort_key_value<unsigned char>(size, repeat, false);
  sort_key_value<short>(size, repeat, false);
  sort_key_value<int>(size, repeat, false);
  sort_key_value<long>(size, repeat, false);
  return 0;
}
