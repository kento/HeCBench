#include <chrono>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <omp.h>

#define BUF_SIZE 256
#define PATTERN  0xDEADBEEF

static void matrixTransposeCPUReference(float *output, const float *input,
                                        unsigned numGroups, unsigned subGroupSize)
{
  for (unsigned i = 0; i < numGroups; ++i)
    for (unsigned j = 0; j < subGroupSize; j++)
      output[i * subGroupSize + j] = input[i * subGroupSize + subGroupSize - j - 1];
}

static void verifyBroadcast(const int *out, int subGroupSize, int pattern = 0)
{
  int expected = pattern;
  if (pattern == 0)
    for (int i = 0; i < subGroupSize; i++) expected += i;

  int errors = 0;
  for (int i = 0; i < BUF_SIZE; i++) {
    if (out[i] != expected) {
      std::cout << "(sg" << subGroupSize << ") ";
      std::cout << "ERROR @ " << i << ":  " << out[i] << "\n";
      ++errors;
      break;
    }
  }
  if (errors == 0) std::cout << "PASS\n";
  else             std::cout << "FAIL\n";
}

static void verifyTransposeMatrix(const float *TransposeMatrix,
                                   const float *cpuTransposeMatrix,
                                   int total, int subGroupSize)
{
  float eps = 1.0e-6f;
  int errors = 0;
  for (int i = 0; i < total; i++) {
    if (std::fabs(TransposeMatrix[i] - cpuTransposeMatrix[i]) > eps) {
      std::cout << "(sg" << subGroupSize << ") ";
      std::cout << "ITEM: " << i
                << " cpu: " << cpuTransposeMatrix[i]
                << " gpu: " << TransposeMatrix[i] << "\n";
      errors++;
      break;
    }
  }
  if (errors == 0) std::cout << "PASS\n";
  else             std::cout << "FAIL\n";
}

int main(int argc, char *argv[])
{
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0]
              << " <repeat for broadcast> <repeat for matrix transpose>\n";
    return 1;
  }
  const int repeat  = atoi(argv[1]);
  const int repeat2 = atoi(argv[2]);

  std::cout << "Broadcast using shuffle functions\n";

  int *d_out = (int*)malloc(BUF_SIZE * sizeof(int));
  #pragma omp target enter data map(alloc: d_out[0:BUF_SIZE])

  // ----------------------------------------------------------------
  // XOR-reduction broadcast (emulated: all threads in a subgroup of
  // size S get sum(0..S-1))
  // ----------------------------------------------------------------
  std::cout << "Broadcast using the shuffle xor function (subgroup sizes 8, 16, and 32) \n";

  auto run_broadcast_xor = [&](int sg) {
    auto begin = std::chrono::steady_clock::now();
    for (int n = 0; n < repeat; n++) {
      #pragma omp target teams distribute parallel for thread_limit(256)
      for (int i = 0; i < BUF_SIZE; i++) {
        int sum = 0;
        for (int j = 0; j < sg; j++) sum += j;
        d_out[i] = sum;
      }
    }
    auto end = std::chrono::steady_clock::now();
    double us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()
                * 1e-3 / repeat;
    std::cout << "Average kernel time (subgroup size = " << sg << "): " << us << " (us)\n";
    #pragma omp target update from(d_out[0:BUF_SIZE])
    verifyBroadcast(d_out, sg);
  };

  run_broadcast_xor(8);
  run_broadcast_xor(16);
  run_broadcast_xor(32);

  // ----------------------------------------------------------------
  // Broadcast from lane 0 (all threads get PATTERN)
  // ----------------------------------------------------------------
  std::cout << "Broadcast using the shuffle function (subgroup sizes 8, 16, and 32) \n";

  auto run_broadcast_const = [&](int sg) {
    auto begin = std::chrono::steady_clock::now();
    for (int n = 0; n < repeat; n++) {
      #pragma omp target teams distribute parallel for thread_limit(256)
      for (int i = 0; i < BUF_SIZE; i++) { d_out[i] = (int)PATTERN; }
    }
    auto end = std::chrono::steady_clock::now();
    double us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()
                * 1e-3 / repeat;
    std::cout << "Average kernel time (subgroup size = " << sg << "): " << us << " (us)\n";
    #pragma omp target update from(d_out[0:BUF_SIZE])
    verifyBroadcast(d_out, sg, (int)PATTERN);
  };

  run_broadcast_const(8);
  run_broadcast_const(16);
  run_broadcast_const(32);

  // ----------------------------------------------------------------
  // Matrix transpose using shuffle (reverse within subgroup)
  // ----------------------------------------------------------------
  std::cout << "matrix transpose using the shuffle function (subgroup sizes are 8, 16, and 32)\n";

  const int total = 1 << 27;

  float *d_in  = (float*)malloc(total * sizeof(float));
  float *d_tpx = (float*)malloc(total * sizeof(float));

  for (int i = 0; i < total; i++) d_in[i] = (float)i * 10.f;

  #pragma omp target enter data map(to: d_in[0:total]) map(alloc: d_tpx[0:total])

  std::vector<float> cpu_tp(total);

  auto run_transpose = [&](int sg) {
    {
      auto begin = std::chrono::steady_clock::now();
      for (int n = 0; n < repeat2; n++) {
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int i = 0; i < total; i++) {
          int group = i / sg;
          int lane  = i % sg;
          d_tpx[group * sg + lane] = d_in[group * sg + sg - lane - 1];
        }
      }
      auto end = std::chrono::steady_clock::now();
      double us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()
                  * 1e-3 / repeat2;
      std::cout << "Average kernel time (subgroup size = " << sg << "): " << us << " (us)\n";
    }

    #pragma omp target update from(d_tpx[0:total])
    matrixTransposeCPUReference(cpu_tp.data(), d_in, total / sg, sg);
    verifyTransposeMatrix(d_tpx, cpu_tp.data(), total, sg);
  };

  run_transpose(8);
  run_transpose(16);
  run_transpose(32);

  #pragma omp target exit data map(delete: d_out[0:BUF_SIZE], d_in[0:total], d_tpx[0:total])
  free(d_out);
  free(d_in);
  free(d_tpx);
  return 0;
}
