#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <omp.h>

#define NUM 4

void reference_kernel(const float* A, unsigned char* out, const unsigned int n)
{
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int idx = 0; idx < (int)(n / 4); idx++) {
    const int base = idx * 4;
    out[base + 0] = (unsigned char)(int)A[base + 0];
    out[base + 1] = (unsigned char)(int)A[base + 1];
    out[base + 2] = (unsigned char)(int)A[base + 2];
    out[base + 3] = (unsigned char)(int)A[base + 3];
  }
}

void blockAccess_kernel(const float* A, unsigned char* out, const unsigned int n)
{
  constexpr int ITEMS_TO_LOAD = 256 * NUM;
  const int num_blocks = ((int)n + ITEMS_TO_LOAD - 1) / ITEMS_TO_LOAD;

  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int bid = 0; bid < num_blocks; bid++) {
    const int base = bid * ITEMS_TO_LOAD;
    const int end  = (base + ITEMS_TO_LOAD < (int)n) ? base + ITEMS_TO_LOAD : (int)n;
    for (int i = base; i < end; i += NUM) {
      float vals[NUM];
      unsigned char qvals[NUM];
      for (int j = 0; j < NUM; j++)
        vals[j] = ((i + j) < end) ? A[i + j] : 0.0f;
      for (int j = 0; j < NUM; j++)
        qvals[j] = (unsigned char)(int)vals[j];
      for (int j = 0; j < NUM; j++)
        if ((i + j) < end) out[i + j] = qvals[j];
    }
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

  const unsigned int n = (unsigned int)nrows * ncols;

  std::vector<float> h_A(n);
  std::mt19937 gen{19937};
  std::normal_distribution<float> dist{128.0f, 127.0f};
  for (unsigned int i = 0; i < n; i++) h_A[i] = dist(gen);

  float*         d_A   = (float*)malloc(n * sizeof(float));
  unsigned char* d_out = (unsigned char*)malloc(n * sizeof(unsigned char));

  for (unsigned int i = 0; i < n; i++) d_A[i] = h_A[i];

  #pragma omp target enter data map(to: d_A[0:n]) map(alloc: d_out[0:n])

  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++)
    reference_kernel(d_A, d_out, n);
  auto end = std::chrono::steady_clock::now();
  long long time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the reference kernel: %f (us)\n",
         (time_ns * 1e-3) / repeat);

  start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++)
    blockAccess_kernel(d_A, d_out, n);
  end = std::chrono::steady_clock::now();
  time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the blockAccess kernel: %f (us)\n",
         (time_ns * 1e-3) / repeat);

  #pragma omp target update from(d_out[0:n])

  bool error = false;
  for (unsigned int i = 0; i < n; i++) {
    unsigned char expected = (unsigned char)(int)h_A[i];
    if (d_out[i] != expected) {
      printf("@%u: %u != %u\n", i, (unsigned)d_out[i], (unsigned)expected);
      error = true;
      break;
    }
  }
  printf("%s\n", error ? "FAIL" : "PASS");

  #pragma omp target exit data map(delete: d_A[0:n], d_out[0:n])
  free(d_A);
  free(d_out);
  return 0;
}
