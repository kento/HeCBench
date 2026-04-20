/*
 * OpenMP target offloading port of atomicSystemWide benchmark.
 */

#include <cmath>
#include <cstdio>
#include <chrono>
#include <cstdint>
#include <omp.h>

#define MYMIN(a,b) ((a) < (b) ? (a) : (b))
#define MYMAX(a,b) ((a) > (b) ? (a) : (b))

int verify(int *atom_arr, const int len, const int loop_num) {
  int val = 0;
  for (int i = 0; i < len * loop_num; i++) val += 10;
  if (val != atom_arr[0]) { printf("atomicAdd failed val=%d data=%d\n", val, atom_arr[0]); return 0; }

  bool found = false;
  for (int i = 0; i < len; i++) if (i == atom_arr[1]) { found = true; break; }
  if (!found) { printf("atomicExch failed\n"); return 0; }

  val = -(1 << 8);
  for (int i = 0; i < len; i++) val = MYMAX(val, i);
  if (val != atom_arr[2]) { printf("atomicMax failed\n"); return 0; }

  val = 1 << 8;
  for (int i = 0; i < len; i++) val = MYMIN(val, i);
  if (val != atom_arr[3]) { printf("atomicMin failed\n"); return 0; }

  found = false;
  for (int i = 0; i < len; i++) if (i == atom_arr[6]) { found = true; break; }
  if (!found) { printf("atomicCAS failed\n"); return 0; }

  val = 0xff;
  for (int i = 0; i < len; i++) val &= (2 * i + 7);
  if (val != atom_arr[7]) { printf("atomicAnd failed\n"); return 0; }

  val = 0;
  for (int i = 0; i < len; i++) val |= (1 << i);
  if (val != atom_arr[8]) { printf("atomicOr failed\n"); return 0; }

  val = 0xff;
  for (int i = 0; i < len; i++) val ^= i;
  if (val != atom_arr[9]) { printf("atomicXor failed\n"); return 0; }

  return 1;
}

int main(int argc, char** argv) {
  if (argc != 2) { printf("Usage: %s <loop count>\n", argv[0]); return 1; }
  const int loop_num = atoi(argv[1]);

  const unsigned int numThreads = 256;
  const unsigned int numBlocks  = 64;
  const unsigned int numData    = 10;
  const int N = numThreads * numBlocks;

  int* h_arr = (int*)calloc(numData, sizeof(int));
  h_arr[7] = h_arr[9] = 0xff;

  #pragma omp target enter data map(to: h_arr[0:numData])

  auto start = std::chrono::steady_clock::now();

  // GPU kernel
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int tid = 0; tid < N; tid++) {
    for (int i = 0; i < loop_num; i++) {
      #pragma omp atomic update
      h_arr[0] += 10;
      #pragma omp atomic write
      h_arr[1] = tid;
      #pragma omp atomic update
      if (h_arr[2] < tid) h_arr[2] = tid;
      #pragma omp atomic update
      if (h_arr[3] > tid) h_arr[3] = tid;
      // CAS: swap h_arr[6] from tid-1 to tid
      {
        int expected = tid - 1;
        __atomic_compare_exchange_n(&h_arr[6], &expected, tid, false,
                                    __ATOMIC_RELAXED, __ATOMIC_RELAXED);
      }
      #pragma omp atomic update
      h_arr[7] &= (2 * tid + 7);
      #pragma omp atomic update
      h_arr[8] |= (1 << tid);
      #pragma omp atomic update
      h_arr[9] ^= tid;
    }
  }

  #pragma omp target update from(h_arr[0:numData])

  // CPU side (simulating the original CPU kernel)
  for (unsigned i = N; i < 2*N; i++) {
    for (int j = 0; j < loop_num; j++) {
      __sync_fetch_and_add(&h_arr[0], 10);
      __sync_lock_test_and_set(&h_arr[1], (int)i);
      int old, exp;
      do { exp = h_arr[2]; old = __sync_val_compare_and_swap(&h_arr[2], exp, MYMAX(exp, (int)i)); } while (old != exp);
      do { exp = h_arr[3]; old = __sync_val_compare_and_swap(&h_arr[3], exp, MYMIN(exp, (int)i)); } while (old != exp);
      __sync_val_compare_and_swap(&h_arr[6], (int)i-1, (int)i);
      __sync_fetch_and_and(&h_arr[7], 2*(int)i+7);
      __sync_fetch_and_or(&h_arr[8], 1<<(int)i);
      __sync_fetch_and_xor(&h_arr[9], (int)i);
    }
  }

  // Re-upload combined results (host side already merged with device results above)
  #pragma omp target update to(h_arr[0:numData])

  auto end = std::chrono::steady_clock::now();
  auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Execution time of atomic kernels on host and device: %f (s)\n", ns * 1e-9f);

  #pragma omp target exit data map(delete: h_arr[0:numData])

  int testResult = verify(h_arr, 2 * N, loop_num);
  printf("systemWideAtomics completed, returned %s\n", testResult ? "OK" : "ERROR!");
  free(h_arr);
  return 0;
}
