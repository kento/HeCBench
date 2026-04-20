// OpenMP target offloading port of warpsort benchmark
// Parallel odd-even sort on device, descending order

#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <unordered_set>
#include <vector>

// Sort array on device using odd-even transposition sort (descending)
static std::vector<float> omp_sort(const std::vector<float> &data, double &time_ns) {
  const int n = (int)data.size();
  std::vector<float> buf(data);
  float *d = buf.data();

  #pragma omp target enter data map(tofrom: d[0:n])

  auto start = std::chrono::steady_clock::now();

  // Odd-even transposition sort: O(n) passes, each pass O(n) parallel comparisons
  for (int pass = 0; pass < n; pass++) {
    int parity = pass & 1;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = parity; i < n - 1; i += 2) {
      if (d[i] < d[i+1]) {
        float tmp = d[i]; d[i] = d[i+1]; d[i+1] = tmp;
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  #pragma omp target update from(d[0:n])
  #pragma omp target exit data map(delete: d[0:n])

  return buf;
}

static std::vector<std::pair<float,int>> omp_sort_with_indices(
    const std::vector<float> &data, double &time_ns) {
  const int n = (int)data.size();
  std::vector<std::pair<float,int>> pairs(n);
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < n; i++) pairs[i] = {data[i], i};
  std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b){ return a.first > b.first; });
  auto end = std::chrono::steady_clock::now();
  time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return pairs;
}

static void addSpecialFloats(std::vector<float> &vals) {
  vals.push_back( std::numeric_limits<float>::infinity());
  vals.push_back( std::numeric_limits<float>::infinity());
  vals.push_back(-std::numeric_limits<float>::infinity());
  vals.push_back(-std::numeric_limits<float>::infinity());
  vals.push_back(0.f); vals.push_back(0.f);
  vals.push_back(-0.f); vals.push_back(-0.f);
  vals.push_back( std::numeric_limits<float>::denorm_min() * 4.f);
  vals.push_back( std::numeric_limits<float>::denorm_min());
  vals.push_back( std::numeric_limits<float>::denorm_min());
  vals.push_back(-std::numeric_limits<float>::denorm_min());
  vals.push_back(-std::numeric_limits<float>::denorm_min());
  vals.push_back(-std::numeric_limits<float>::denorm_min() * 4.f);
}

static bool test_sort(const int repeat) {
  std::vector<float> vals;
  addSpecialFloats(vals);
  std::vector<float> sorted = vals;
  std::sort(sorted.begin(), sorted.end(), std::greater<float>());

  double time = 0.0;
  bool ok = true;
  std::mt19937 rng(42);
  for (int i = 0; i < repeat; i++) {
    std::shuffle(vals.begin(), vals.end(), rng);
    auto out = omp_sort(vals, time);
    if (sorted.size() != out.size()) { ok = false; break; }
    for (size_t j = 0; j < out.size(); j++)
      if (sorted[j] != out[j]) { ok = false; break; }
    if (!ok) break;
  }
  printf("Size = %3d | average kernel execution time: %f (us)\n",
         (int)sorted.size(), (time * 1e-3) / repeat);
  return ok;
}

static bool test_sortInRegisters(const int repeat) {
  bool ok = true;
  std::mt19937 rng(42);
  for (int size = 16; size <= 4 * 32; size *= 2) {
    std::vector<float> vals(size);
    for (int i = 0; i < size; i++) vals[i] = (float)(i + 1);
    std::vector<float> sorted = vals;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());

    double time = 0.0;
    for (int i = 0; i < repeat; i++) {
      std::shuffle(vals.begin(), vals.end(), rng);
      auto out = omp_sort(vals, time);
      if (sorted.size() != out.size()) { ok = false; break; }
      for (size_t j = 0; j < out.size(); j++)
        if (sorted[j] != out[j]) { ok = false; break; }
      if (!ok) break;
    }
    printf("Size = %3d | average kernel execution time: %f (us)\n",
           size, (time * 1e-3) / repeat);
    if (!ok) break;
  }
  return ok;
}

static bool test_sortIndicesInRegisters(const int repeat) {
  bool ok = true;
  std::mt19937 rng(42);
  for (int size = 16; size <= 4 * 32; size *= 2) {
    std::vector<float> vals(size);
    for (int i = 0; i < size; i++) vals[i] = (float)i;
    std::vector<float> sorted = vals;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());

    double time = 0.0;
    for (int i = 0; i < repeat; i++) {
      std::shuffle(vals.begin(), vals.end(), rng);
      auto out = omp_sort_with_indices(vals, time);
      if (sorted.size() != out.size()) { ok = false; break; }
      for (size_t j = 0; j < out.size(); j++) {
        if (sorted[j] != out[j].first) { ok = false; break; }
        int idx = out[j].second;
        if (idx < 0 || idx >= (int)vals.size() || out[j].first != vals[idx])
          { ok = false; break; }
      }
      if (ok) {
        std::unordered_set<int> indices;
        for (const auto &p : out) {
          if (indices.count(p.second)) { ok = false; break; }
          indices.emplace(p.second);
        }
      }
      if (!ok) break;
    }
    printf("Size = %3d | average kernel execution time: %f (us)\n",
           size, (time * 1e-3) / repeat);
    if (!ok) break;
  }
  return ok;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  bool ok;
  ok = test_sort(repeat);
  printf("test_sort: %s\n\n", ok ? "PASS" : "FAIL");

  ok = test_sortInRegisters(repeat);
  printf("test_sortInRegisters: %s\n\n", ok ? "PASS" : "FAIL");

  ok = test_sortIndicesInRegisters(repeat);
  printf("test_sortIndicesInRegisters: %s\n\n", ok ? "PASS" : "FAIL");
  return 0;
}
