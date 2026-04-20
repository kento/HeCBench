// OpenMP target port of sc (stream compaction) benchmark.
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <chrono>
#include <vector>

int main(int argc, char* argv[]) {
  int n = 8388608;
  int reps = 100;
  int warmup = 5;
  int compaction_factor = 50;
  int remove_value = 0;

  int opt;
  while ((opt = getopt(argc, argv, "n:r:w:c:")) >= 0) {
    switch (opt) {
      case 'n': n = atoi(optarg); break;
      case 'r': reps = atoi(optarg); break;
      case 'w': warmup = atoi(optarg); break;
      case 'c': compaction_factor = atoi(optarg); break;
      default: break;
    }
  }

  const int keep_count = (int)((long long)n * compaction_factor / 100);

  std::vector<int> h_input(n, remove_value);
  srand(42);
  std::vector<int> positions(n);
  for (int i = 0; i < n; i++) positions[i] = i;
  for (int i = 0; i < keep_count; i++) {
    int j = i + rand() % (n - i);
    std::swap(positions[i], positions[j]);
    h_input[positions[i]] = (rand() % 254) + 1;
  }

  int* d_input  = (int*) malloc(n * sizeof(int));
  int* d_flags  = (int*) malloc(n * sizeof(int));
  int* d_scan   = (int*) malloc(n * sizeof(int));
  int* d_output = (int*) malloc(n * sizeof(int));

  memcpy(d_input, h_input.data(), n * sizeof(int));

  #pragma omp target enter data map(to: d_input[0:n]) \
                                map(alloc: d_flags[0:n], d_scan[0:n], d_output[0:n])

  const int rv = remove_value;
  double total_ms = 0.0;

  for (int iter = 0; iter < warmup + reps; iter++) {
    auto t0 = std::chrono::steady_clock::now();

    // Compute flags
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < n; i++)
      d_flags[i] = (d_input[i] != rv) ? 1 : 0;

    // Exclusive prefix scan using OpenMP inscan reduction
    {
      int prefix = 0;
      #pragma omp target teams distribute parallel for \
              reduction(inscan, +:prefix) thread_limit(256)
      for (int i = 0; i < n; i++) {
        d_scan[i] = prefix;
        #pragma omp scan exclusive(prefix)
        prefix += d_flags[i];
      }
    }

    // Scatter
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < n; i++) {
      if (d_flags[i]) d_output[d_scan[i]] = d_input[i];
    }

    auto t1 = std::chrono::steady_clock::now();
    if (iter >= warmup)
      total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  printf("Total stream compaction time for %d iterations: %f (ms)\n", reps, total_ms);

  // Verification
  #pragma omp target update from(d_scan[0:n], d_flags[0:n])
  int output_count = d_scan[n - 1] + d_flags[n - 1];

  if (output_count == keep_count)
    printf("Verification: PASS\n");
  else
    printf("Verification: FAIL (got %d, expected %d)\n", output_count, keep_count);

  #pragma omp target exit data map(delete: d_input[0:n], d_flags[0:n], d_scan[0:n], d_output[0:n])

  free(d_input); free(d_flags); free(d_scan); free(d_output);
  return 0;
}
