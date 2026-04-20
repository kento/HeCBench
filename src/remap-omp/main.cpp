// OpenMP target offloading port of remap benchmark.
// Remaps integer values to dense 0..K-1 range using sort + unique.
// Host-side sort/unique algorithms; device kernel for the remap scatter.

#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <vector>

void eval_remap(const int N, const int repeat) {
  std::vector<int> h_input(N);
  std::vector<int> h_output(N);

  srand(123);
  for (int i = 0; i < N; i++)
    h_input[i] = rand() % N;

  long seq_time = 0, sort_time = 0, unique_time = 0,
       kernel_time = 0, copy_time = 0, alloc_time = 0;

  auto offload_start = std::chrono::steady_clock::now();

  for (int rep = 0; rep < repeat; rep++) {
    auto t0 = std::chrono::steady_clock::now();
    int *d_input       = new int[N];
    int *d_output      = new int[N];
    int *d_first_order = new int[N];
    int *d_second_order= new int[N];
    auto t1 = std::chrono::steady_clock::now();
    alloc_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; i++) d_input[i] = h_input[i];
#pragma omp target enter data map(to: d_input[0:N]) \
                              map(alloc: d_output[0:N], d_first_order[0:N], d_second_order[0:N])
    t1 = std::chrono::steady_clock::now();
    copy_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    t0 = std::chrono::steady_clock::now();
    std::vector<int> h_order1(N);
    std::iota(h_order1.begin(), h_order1.end(), 0);
    t1 = std::chrono::steady_clock::now();
    seq_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    t0 = std::chrono::steady_clock::now();
    std::stable_sort(h_order1.begin(), h_order1.end(),
                     [&](int a, int b) { return h_input[a] < h_input[b]; });
    std::vector<int> h_sorted(N);
    for (int i = 0; i < N; i++) h_sorted[i] = h_input[h_order1[i]];
    t1 = std::chrono::steady_clock::now();
    sort_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    t0 = std::chrono::steady_clock::now();
    std::vector<int> h_second_order;
    h_second_order.reserve(N);
    for (int i = 0; i < N; i++)
      if (i == 0 || h_sorted[i] != h_sorted[i - 1])
        h_second_order.push_back(i);
    int K = static_cast<int>(h_second_order.size());
    t1 = std::chrono::steady_clock::now();
    unique_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    if (rep == 0)
      printf("Percentage of unique elements: %.1f %%\n", (float)K * 100.f / N);

    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; i++) d_first_order[i] = h_order1[i];
    for (int i = 0; i < K; i++) d_second_order[i] = h_second_order[i];
#pragma omp target update to(d_first_order[0:N], d_second_order[0:K])
    t1 = std::chrono::steady_clock::now();
    copy_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    t0 = std::chrono::steady_clock::now();
#pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < K; i++) {
      int idx = d_second_order[i];
      d_output[d_first_order[idx]] = i;
      for (idx++; idx < N && (i == K - 1 || idx != d_second_order[i + 1]); idx++)
        d_output[d_first_order[idx]] = i;
    }
    t1 = std::chrono::steady_clock::now();
    kernel_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    t0 = std::chrono::steady_clock::now();
#pragma omp target update from(d_output[0:N])
    for (int i = 0; i < N; i++) h_output[i] = d_output[i];
    t1 = std::chrono::steady_clock::now();
    copy_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

#pragma omp target exit data map(delete: d_input[0:N], d_output[0:N], \
                                         d_first_order[0:N], d_second_order[0:N])
    delete[] d_input; delete[] d_output; delete[] d_first_order; delete[] d_second_order;
  }

  auto offload_end = std::chrono::steady_clock::now();
  auto offload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
      offload_end - offload_start).count();

  printf("Average offload time: %f (s)\n", offload_time * 1e-9f / repeat);
  printf("Average execution time of memory allocation : %f (us)\n", (alloc_time * 1e-3f) / repeat);
  printf("Average execution time of data copy : %f (us)\n", (copy_time * 1e-3f) / repeat);
  printf("Average execution time of sequence : %f (us)\n", (seq_time * 1e-3f) / repeat);
  printf("Average execution time of sort-by-key : %f (us)\n", (sort_time * 1e-3f) / repeat);
  printf("Average execution time of unique-by-key : %f (us)\n", (unique_time * 1e-3f) / repeat);
  printf("Average execution time of remap kernel: %f (us)\n", (kernel_time * 1e-3f) / repeat);

  int cs1 = 0, cs2 = 0;
  for (int i = 0; i < N - 1; i++) cs1 ^= h_output[i] - h_output[i + 1];
  for (int i = 0; i < N;     i++) cs2 ^= h_output[i];
  printf("Checksum: %d %d\n", cs1, cs2);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int N      = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  for (int i = 0; i < 2; i++) {
    printf("\nRun %d\n", i);
    eval_remap(N, repeat);
  }
  return 0;
}
