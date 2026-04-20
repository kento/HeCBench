#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

// Binary predicate
#pragma omp declare target
bool valid(int x) { return x > 0; }
#pragma omp end declare target

// Run an exclusive prefix scan over N elements using a single device thread.
// For small N (32..1024 as in this benchmark), this is simple and correct.
template <int N>
void bscan(const int repeat)
{
  int* d_in  = (int*)malloc(N * sizeof(int));
  int* d_out = (int*)malloc(N * sizeof(int));

  int  h_in_arr[N], h_out_arr[N], ref_out[N];
  bool ok          = true;
  double total_ns  = 0.0;
  int    valid_count = 0;

  constexpr size_t grid_size = 12ULL * 7 * 8 * 9 * 10;

  srand(123);

  #pragma omp target enter data map(alloc: d_in[0:N], d_out[0:N])

  for (int iter = 0; iter < repeat; iter++) {
    for (int i = 0; i < N; i++) {
      h_in_arr[i] = rand() % N - N / 2;
      if (valid(h_in_arr[i])) valid_count++;
      d_in[i] = h_in_arr[i];
    }
    #pragma omp target update to(d_in[0:N])

    auto t0 = std::chrono::steady_clock::now();

    // Sequential exclusive prefix scan on device
    #pragma omp target
    {
      int prefix = 0;
      for (int i = 0; i < N; i++) {
        d_out[i] = prefix;
        if (valid(d_in[i])) prefix++;
      }
    }

    auto t1 = std::chrono::steady_clock::now();
    total_ns += (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    #pragma omp target update from(d_out[0:N])
    for (int i = 0; i < N; i++) h_out_arr[i] = d_out[i];

    ref_out[0] = 0;
    ok &= (h_out_arr[0] == ref_out[0]);
    for (int i = 1; i < N; i++) {
      ref_out[i] = ref_out[i - 1] + (h_in_arr[i - 1] > 0 ? 1 : 0);
      ok &= (ref_out[i] == h_out_arr[i]);
    }
    if (!ok) break;
  }

  printf("Block size = %d, ratio of valid elements = %f, verify = %s\n",
         N, valid_count * 1.f / (N * repeat), ok ? "PASS" : "FAIL");

  if (ok) {
    printf("Average execution time: %f (us)\n", (total_ns * 1e-3) / repeat);
    printf("Billion elements per second: %f\n\n",
           (double)grid_size * N * repeat / total_ns);
  }

  #pragma omp target exit data map(delete: d_in[0:N], d_out[0:N])
  free(d_in);
  free(d_out);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  bscan<32>(repeat);
  bscan<64>(repeat);
  bscan<128>(repeat);
  bscan<256>(repeat);
  bscan<512>(repeat);
  bscan<1024>(repeat);
  return 0;
}
