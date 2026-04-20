// GPU hash map benchmark – OpenMP target offloading port
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>

static constexpr uint32_t EMPTY_KEY   = 0xFFFFFFFFu;
static constexpr uint32_t EMPTY_VALUE = 0u;

#pragma omp declare target
uint32_t hash_fn(uint32_t key, uint32_t table_size) {
    return key % table_size;
}
#pragma omp end declare target

void hash_insert(
    const uint32_t* d_keys,
    const uint32_t* d_values,
    uint32_t* tbl_keys,
    uint32_t* tbl_values,
    uint32_t numKeys,
    uint32_t table_size)
{
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < (int)numKeys; i++) {
    uint32_t key = d_keys[i];
    uint32_t val = d_values[i];
    uint32_t h   = hash_fn(key, table_size);
    bool inserted = false;
    while (!inserted) {
      uint32_t old = EMPTY_KEY;
      #pragma omp atomic compare capture
      { old = tbl_keys[h]; if (tbl_keys[h] == EMPTY_KEY) tbl_keys[h] = key; }
      if (old == EMPTY_KEY || old == key) {
        tbl_values[h] = val;
        inserted = true;
      } else {
        h = (h + 1) % table_size;
      }
    }
  }
}

void hash_search(
    const uint32_t* d_queries,
    const uint32_t* tbl_keys,
    const uint32_t* tbl_values,
    uint32_t* d_results,
    uint32_t numQueries,
    uint32_t table_size)
{
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < (int)numQueries; i++) {
    uint32_t key = d_queries[i];
    uint32_t h   = hash_fn(key, table_size);
    bool found = false;
    while (!found) {
      if (tbl_keys[h] == key) {
        d_results[i] = tbl_values[h];
        found = true;
      } else if (tbl_keys[h] == EMPTY_KEY) {
        d_results[i] = EMPTY_VALUE;
        found = true;
      } else {
        h = (h + 1) % table_size;
      }
    }
  }
}

int main(int argc, char* argv[]) {
  uint32_t numKeys    = 1u << 20;
  uint32_t numQueries = numKeys;
  if (argc > 1) numKeys    = (uint32_t)std::atoi(argv[1]);
  if (argc > 2) numQueries = (uint32_t)std::atoi(argv[2]);

  printf("Building the table with %u keys\n",    numKeys);
  printf("Searching the table with %u queries\n", numQueries);

  std::mt19937 rng(42);

  std::vector<uint32_t> h_keys(numKeys);
  std::iota(h_keys.begin(), h_keys.end(), 0u);
  std::shuffle(h_keys.begin(), h_keys.end(), rng);

  std::vector<uint32_t> h_values(numKeys);
  for (uint32_t i = 0; i < numKeys; ++i) h_values[i] = h_keys[i];

  std::vector<uint32_t> h_queries(numQueries);
  {
    std::vector<uint32_t> pool(numQueries * 2);
    std::iota(pool.begin(), pool.end(), 0u);
    std::shuffle(pool.begin(), pool.end(), rng);
    std::copy(pool.begin(), pool.begin() + numQueries, h_queries.begin());
  }

  const uint32_t table_size = numKeys * 2;

  uint32_t* d_keys     = (uint32_t*)malloc(numKeys    * sizeof(uint32_t));
  uint32_t* d_values   = (uint32_t*)malloc(numKeys    * sizeof(uint32_t));
  uint32_t* d_queries  = (uint32_t*)malloc(numQueries * sizeof(uint32_t));
  uint32_t* d_results  = (uint32_t*)malloc(numQueries * sizeof(uint32_t));
  uint32_t* tbl_keys   = (uint32_t*)malloc(table_size * sizeof(uint32_t));
  uint32_t* tbl_values = (uint32_t*)malloc(table_size * sizeof(uint32_t));

  for (uint32_t i = 0; i < numKeys;    ++i) { d_keys[i] = h_keys[i]; d_values[i] = h_values[i]; }
  for (uint32_t i = 0; i < numQueries; ++i)   d_queries[i] = h_queries[i];
  for (uint32_t i = 0; i < table_size; ++i) { tbl_keys[i] = EMPTY_KEY; tbl_values[i] = EMPTY_VALUE; }

  #pragma omp target enter data map(alloc: d_keys[0:numKeys], d_values[0:numKeys], \
      d_queries[0:numQueries], d_results[0:numQueries], \
      tbl_keys[0:table_size], tbl_values[0:table_size])
  #pragma omp target update to(d_keys[0:numKeys], d_values[0:numKeys], \
      d_queries[0:numQueries], tbl_keys[0:table_size], tbl_values[0:table_size])

  auto t_start = std::chrono::high_resolution_clock::now();
  hash_insert(d_keys, d_values, tbl_keys, tbl_values, numKeys, table_size);
  auto t_build = std::chrono::high_resolution_clock::now();
  hash_search(d_queries, tbl_keys, tbl_values, d_results, numQueries, table_size);
  auto t_end = std::chrono::high_resolution_clock::now();

  double build_ms  = std::chrono::duration_cast<std::chrono::microseconds>(t_build - t_start).count() / 1e3;
  double search_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end   - t_build).count() / 1e3;
  printf("Build  time: %.3f ms\n",  build_ms);
  printf("Search time: %.3f ms\n",  search_ms);
  printf("Total  time: %.3f ms\n",  build_ms + search_ms);

  #pragma omp target update from(d_results[0:numQueries])
  int errors = 0;
  for (uint32_t i = 0; i < numQueries; ++i) {
    uint32_t q = h_queries[i];
    if (q < numKeys) {
      if (d_results[i] != q) ++errors;
    } else {
      if (d_results[i] != 0u) ++errors;
    }
  }
  printf("%s\n", (errors == 0) ? "PASS" : "VERIFICATION FAILED");

  #pragma omp target exit data map(delete: d_keys[0:numKeys], d_values[0:numKeys], \
      d_queries[0:numQueries], d_results[0:numQueries], \
      tbl_keys[0:table_size], tbl_values[0:table_size])
  free(d_keys); free(d_values); free(d_queries); free(d_results);
  free(tbl_keys); free(tbl_values);
  return 0;
}
