// Collision detection benchmark – OpenMP target offloading port
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <chrono>
#include <vector>
#include <omp.h>

#pragma omp declare target
void insertion_sort(int* arr, int n) {
  for (int i = 1; i < n; ++i) {
    int key = arr[i], j = i - 1;
    while (j >= 0 && arr[j] > key) {
      arr[j + 1] = arr[j];
      --j;
    }
    arr[j + 1] = key;
  }
}

bool warpHasCollision(const int* vals, int n) {
  int sorted[32];
  for (int i = 0; i < n; ++i) sorted[i] = vals[i];
  insertion_sort(sorted, n);
  for (int i = 1; i < n; ++i)
    if (sorted[i] == sorted[i - 1]) return true;
  return false;
}

unsigned int warpCollisionMask(const int* vals, int n) {
  unsigned int mask = 0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      if (i != j && vals[i] == vals[j]) {
        mask |= (1u << i);
        break;
      }
  return mask;
}
#pragma omp end declare target

static int d_result_val;
static unsigned int d_mask_val;

void checkDuplicatesOMP(const int* h_v, int n)
{
  int* d_v = (int*)malloc(n * sizeof(int));
  for (int i = 0; i < n; ++i) d_v[i] = h_v[i];
  int result = 0;
  #pragma omp target enter data map(alloc: d_v[0:n])
  #pragma omp target update to(d_v[0:n])
  #pragma omp target map(tofrom: result)
  {
    result = warpHasCollision(d_v, n) ? 1 : 0;
  }
  d_result_val = result;
  #pragma omp target exit data map(delete: d_v[0:n])
  free(d_v);
}

void checkDuplicateMaskOMP(const int* h_v, int n)
{
  int* d_v = (int*)malloc(n * sizeof(int));
  for (int i = 0; i < n; ++i) d_v[i] = h_v[i];
  unsigned int mask = 0;
  #pragma omp target enter data map(alloc: d_v[0:n])
  #pragma omp target update to(d_v[0:n])
  #pragma omp target map(tofrom: mask)
  {
    mask = warpCollisionMask(d_v, n);
  }
  d_mask_val = mask;
  #pragma omp target exit data map(delete: d_v[0:n])
  free(d_v);
}

static void build_test_vector(int ND, int numDups, std::vector<int>& v) {
  v.clear();
  for (int i = 0; i < ND - numDups; ++i) {
    int r = 0;
    while (true) {
      r = rand();
      bool found = false;
      for (int x : v) if (x == r) { found = true; break; }
      if (!found) break;
    }
    v.push_back(r);
  }
  for (int i = 0; i < numDups; ++i) v.push_back(v[0]);
}

void test_collision(int ND)
{
  for (int numDups = 0; numDups < ND; ++numDups) {
    std::vector<int> vec;
    build_test_vector(ND, numDups, vec);
    checkDuplicatesOMP(vec.data(), ND);
    bool detected = (d_result_val != 0);
    bool expected = (numDups > 0);
    assert(detected == expected);
    (void)detected; (void)expected;
  }
}

void test_collisionMask(int ND)
{
  for (int numDups = 0; numDups < ND; ++numDups) {
    std::vector<int> vec;
    build_test_vector(ND, numDups, vec);
    checkDuplicateMaskOMP(vec.data(), ND);
    unsigned int mask = d_mask_val;
    bool hasDup   = (mask != 0);
    bool expected = (numDups > 0);
    if (hasDup != expected) {
      printf("Error: numDups=%d expected=%x mask=%x\n", numDups, expected ? 0xffffffffU : 0, mask);
      break;
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  srand(123);
  const int num_dup = 32;
  const int repeat  = atoi(argv[1]);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i)
    test_collision(num_dup);
  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the function test_collision: %f (us)\n",
         time * 1e-3f / repeat);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i)
    test_collisionMask(num_dup);
  end  = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the function test_collisionMask: %f (us)\n",
         time * 1e-3f / repeat);
  return 0;
}
