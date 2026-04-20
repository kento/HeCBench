// Port of nosync-cuda to OpenMP target offloading.
// Tests that reduce_result - last_scan_element == 0.
// For [0,1,...,n-1]: reduce = n*(n-1)/2 and last inclusive_scan element = n*(n-1)/2.
#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>

int main(int argc, char *argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]), repeat = atoi(argv[2]);

  int sum = -1;
  auto start = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    int *d_vec = (int *)malloc(n * sizeof(int));
    int *d_res = (int *)malloc(n * sizeof(int));
#pragma omp target enter data map(alloc: d_vec[0:n], d_res[0:n])

    // Fill d_vec[i] = i on device.
#pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < n; i++) d_vec[i] = i;

    // Inclusive prefix scan: download d_vec, scan on host, upload d_res.
#pragma omp target update from(d_vec[0:n])
    {
      int running = 0;
      for (int i = 0; i < n; i++) {
        running += d_vec[i];
        d_res[i] = running;
      }
    }
#pragma omp target update to(d_res[0:n])

    // Parallel reduce on device.
    int reduce_result = 0;
#pragma omp target teams distribute parallel for reduction(+:reduce_result) thread_limit(256)
    for (int i = 0; i < n; i++) reduce_result += d_vec[i];

    // Read last scan element.
#pragma omp target update from(d_res[0:n])
    int last_scan = d_res[n - 1];
    sum = reduce_result - last_scan;

#pragma omp target exit data map(delete: d_vec[0:n], d_res[0:n])
    free(d_vec);
    free(d_res);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time: " << (time * 1e-3f) / repeat << " (us)\n";
  std::cout << ((sum == 0) ? "PASS" : "FAIL") << "\n";
  return 0;
}
