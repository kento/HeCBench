#include "utils.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <chrono>
#include <vector>
#include <limits>
#include <algorithm>

static void runBitPackingOnGPU(
    int32_t const* inputHost,
    void*          outputHost,
    int const      numBitsMax,
    size_t const   n,
    int*           numBitsOut,
    int32_t*       minValOut)
{
  const size_t packedSizeBytes =
      (((static_cast<size_t>(numBitsMax) * n) / 64U) + 1U) * 8U;
  const size_t packedSizeWords =
      (packedSizeBytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);

  int32_t*       d_input  = (int32_t*)malloc(n * sizeof(int32_t));
  uint32_t*      d_output = (uint32_t*)malloc(packedSizeWords * sizeof(uint32_t));
  int32_t        h_minValue  = 0;
  unsigned char  h_numBits   = 0;

  memcpy(d_input, inputHost, n * sizeof(int32_t));
  memset(d_output, 0, packedSizeWords * sizeof(uint32_t));

  #pragma omp target enter data \
    map(to:   d_input[0:n]) \
    map(alloc: d_output[0:packedSizeWords])

  compress(d_output, d_input, n, &h_minValue, &h_numBits);

  #pragma omp target exit data \
    map(from: d_output[0:packedSizeWords]) \
    map(delete: d_input[0:n])

  *minValOut  = h_minValue;
  *numBitsOut = static_cast<int>(h_numBits);

  const size_t copyWords =
      (std::min(packedSizeBytes, n * sizeof(int32_t)) + sizeof(uint32_t) - 1)
      / sizeof(uint32_t);
  memcpy(outputHost, d_output, copyWords * sizeof(uint32_t));

  free(d_input);
  free(d_output);
}

int main(int argc, char* argv[])
{
  const int offset  = 87231;
  const int numBits = 13;

  std::vector<size_t> const sizes{2, 123, 3411, 83621, 872163, 100000001};

  using T = int32_t;

  std::vector<T> source(sizes.back());
  std::srand(0);
  for (T& v : source)
    v = std::abs(static_cast<T>(std::rand())) % std::numeric_limits<T>::max();

  const size_t numBytes = sizes.back() * sizeof(T);
  T*    inputHost  = static_cast<T*>(aligned_alloc(1024, numBytes));
  void* outputHost = aligned_alloc(1024, numBytes);

  for (size_t const n : sizes) {
    for (size_t i = 0; i < n; ++i)
      inputHost[i] =
          (source[i] & ((1U << numBits) - 1)) + offset;

    T   minValue;
    int numBitsAct;

    printf("Size = %10zu\n", n);
    auto start = std::chrono::steady_clock::now();

    runBitPackingOnGPU(inputHost, outputHost, numBits, n, &numBitsAct, &minValue);

    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    end - start).count();
    printf("Device offload time = %f (s)\n", time * 1e-9f);

    assert(numBitsAct <= numBits);

    std::vector<T> unpackedHost;
    unpackedHost.reserve(n);
    for (size_t i = 0; i < n; ++i)
      unpackedHost.emplace_back(unpackBytes(
          outputHost, static_cast<uint8_t>(numBitsAct), minValue, i));

    assert(unpackedHost.size() == n);

    bool ok = true;
    const size_t numSamples =
        static_cast<size_t>(std::sqrt(static_cast<double>(n))) + 1;
    for (size_t i = 0; i < numSamples; ++i) {
      const size_t idx = static_cast<uint32_t>(source[i]) % n;
      if (unpackedHost[idx] != inputHost[idx]) {
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
  }

  free(inputHost);
  free(outputHost);
  return 0;
}
