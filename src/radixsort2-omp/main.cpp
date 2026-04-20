// OpenMP target offloading port of radixsort2 benchmark.
// Keys-only and key-value sort using std::sort (host-side).
// The Kokkos version also used host-side sort for key-value and Kokkos::sort
// (which wraps std::sort on CPU) for keys-only, so this is equivalent.

#include <omp.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static bool hasCmdFlag(int argc, char** argv, const char* flag)
{
  for (int i = 1; i < argc; ++i)
    if (strcmp(argv[i], flag) == 0)
      return true;
  return false;
}

static int getCmdInt(int argc, char** argv, const char* flag, int def)
{
  for (int i = 1; i + 1 < argc; ++i)
    if (strcmp(argv[i], flag) == 0)
      return atoi(argv[i + 1]);
  return def;
}

template <typename T>
static bool keysOnlySort(int numElements, int numIterations, bool floatKeys)
{
  printf("\nSorting %d %s keys (keys-only)\n",
         numElements, floatKeys ? "float" : "unsigned int");

  std::mt19937 rng(19937);
  std::vector<T> h_keys(numElements);

  if (floatKeys) {
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (auto& k : h_keys) k = dist(rng);
  } else {
    std::uniform_int_distribution<unsigned int> dist(0, UINT_MAX);
    for (auto& k : h_keys) k = (T)dist(rng);
  }

  // Allocate device buffer
  T* d_keys = new T[numElements];
  memcpy(d_keys, h_keys.data(), numElements * sizeof(T));
#pragma omp target enter data map(to: d_keys[0:numElements])

  double totalTime = 0.0;

  for (int iter = 0; iter < numIterations; ++iter) {
    // Reset to unsorted
    memcpy(d_keys, h_keys.data(), numElements * sizeof(T));
#pragma omp target update to(d_keys[0:numElements])

    // Copy to host for sort (std::sort on host; update back)
#pragma omp target update from(d_keys[0:numElements])

    auto t0 = std::chrono::steady_clock::now();
    std::sort(d_keys, d_keys + numElements);
    auto t1 = std::chrono::steady_clock::now();

    totalTime += std::chrono::duration<double>(t1 - t0).count();

#pragma omp target update to(d_keys[0:numElements])
  }

  double avgTime = totalTime / numIterations;
  printf("Throughput = %.4f MElements/s, Time = %.5f s\n",
         1.0e-6 * numElements / avgTime, avgTime);

#pragma omp target update from(d_keys[0:numElements])
  bool ok = true;
  for (int i = 1; i < numElements; ++i)
    if (d_keys[i] < d_keys[i - 1]) { ok = false; break; }

#pragma omp target exit data map(delete: d_keys[0:numElements])
  delete[] d_keys;
  return ok;
}

template <typename T>
static bool keyValueSort(int numElements, int numIterations, bool floatKeys)
{
  printf("\nSorting %d %s keys with values (host std::sort)\n",
         numElements, floatKeys ? "float" : "unsigned int");

  std::mt19937 rng(19937);
  std::vector<T> h_keys_orig(numElements);
  if (floatKeys) {
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (auto& k : h_keys_orig) k = dist(rng);
  } else {
    std::uniform_int_distribution<unsigned int> dist(0, UINT_MAX);
    for (auto& k : h_keys_orig) k = (T)dist(rng);
  }

  std::vector<unsigned int> h_values_orig(numElements);
  for (int i = 0; i < numElements; ++i)
    h_values_orig[i] = (unsigned int)i;

  using Pair = std::pair<T, unsigned int>;
  std::vector<Pair> pairs(numElements);

  double totalTime = 0.0;

  for (int iter = 0; iter < numIterations; ++iter) {
    for (int i = 0; i < numElements; ++i)
      pairs[i] = {h_keys_orig[i], h_values_orig[i]};

    auto t0 = std::chrono::steady_clock::now();
    std::sort(pairs.begin(), pairs.end(),
              [](const Pair& a, const Pair& b) { return a.first < b.first; });
    auto t1 = std::chrono::steady_clock::now();
    totalTime += std::chrono::duration<double>(t1 - t0).count();
  }

  double avgTime = totalTime / numIterations;
  printf("Throughput = %.4f MElements/s, Time = %.5f s\n",
         1.0e-6 * numElements / avgTime, avgTime);

  for (int i = 1; i < numElements; ++i)
    if (pairs[i].first < pairs[i - 1].first)
      return false;
  return true;
}

int main(int argc, char** argv)
{
  printf("%s Starting...\n\n", argv[0]);

  const int  numElements = getCmdInt(argc, argv, "-n",          1048576);
  const int  numIter     = getCmdInt(argc, argv, "-iterations", 100);
  const bool floatKeys   = hasCmdFlag(argc, argv, "-float");
  const bool keysOnly    = hasCmdFlag(argc, argv, "-keysonly");

  bool ok = true;
  if (floatKeys) {
    ok &= keysOnlySort<float>(numElements, numIter, true);
    if (!keysOnly)
      ok &= keyValueSort<float>(numElements, numIter, true);
  } else {
    ok &= keysOnlySort<unsigned int>(numElements, numIter, false);
    if (!keysOnly)
      ok &= keyValueSort<unsigned int>(numElements, numIter, false);
  }

  printf("\n%s\n", ok ? "PASS" : "FAIL");
  return 0;
}
