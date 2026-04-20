// OpenMP target port of scan3 benchmark (exclusive prefix scan over float).
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <chrono>
#include <vector>

bool isPowerOf2(int n) { return n > 0 && (n & (n - 1)) == 0; }
int roundToPowerOf2(int n) {
  int p = 1;
  while (p < n) p <<= 1;
  return p;
}

void scanCPUReference(float* output, const float* input, unsigned int length) {
  output[0] = 0.0f;
  for (unsigned int i = 1; i < length; i++)
    output[i] = input[i-1] + output[i-1];
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <repeat> <input_length>\n";
    return 1;
  }
  int iterations = atoi(argv[1]);
  int length = atoi(argv[2]);

  if (iterations < 1) {
    std::cout << "Error, iterations cannot be 0 or negative. Exiting..\n";
    return -1;
  }
  if (!isPowerOf2(length))
    length = roundToPowerOf2(length);

  std::vector<float> input(length);
  std::vector<float> output(length, 0.0f);
  std::vector<float> refOutput(length, 0.0f);

  srand(42);
  for (int i = 0; i < length; i++) input[i] = (float)(rand() % 256);

  scanCPUReference(refOutput.data(), input.data(), length);

  float* d_input  = (float*) malloc(length * sizeof(float));
  float* d_output = (float*) malloc(length * sizeof(float));
  for (int i = 0; i < length; i++) d_input[i] = input[i];

  #pragma omp target enter data map(to: d_input[0:length]) \
                                map(alloc: d_output[0:length])

  std::cout << "Executing kernel for " << iterations << " iterations\n";
  std::cout << "-------------------------------------------\n";

  // Warmup
  {
    float prefix = 0.0f;
    #pragma omp target teams distribute parallel for \
            reduction(inscan, +:prefix) thread_limit(256)
    for (int i = 0; i < length; i++) {
      d_output[i] = prefix;
      #pragma omp scan exclusive(prefix)
      prefix += d_input[i];
    }
  }

  auto t0 = std::chrono::steady_clock::now();
  for (int n = 0; n < iterations; n++) {
    float prefix = 0.0f;
    #pragma omp target teams distribute parallel for \
            reduction(inscan, +:prefix) thread_limit(256)
    for (int i = 0; i < length; i++) {
      d_output[i] = prefix;
      #pragma omp scan exclusive(prefix)
      prefix += d_input[i];
    }
  }
  auto t1 = std::chrono::steady_clock::now();

  double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
  std::cout << "Average execution time of OMP exclusive scan: "
            << ns / iterations * 1e-3 << " (us)\n";

  #pragma omp target update from(d_output[0:length])
  #pragma omp target exit data map(delete: d_input[0:length], d_output[0:length])

  bool pass = true;
  for (int i = 0; i < length; i++) {
    if (fabsf(d_output[i] - refOutput[i]) > 0.001f) { pass = false; break; }
  }
  std::cout << (pass ? "PASS" : "FAIL") << std::endl;

  free(d_input);
  free(d_output);
  return 0;
}
