#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <float.h>
#include <string.h>
#include <chrono>
#include <omp.h>

#define NUM_BLOCKS 1024
#define BLOCK_SIZE 256

// CAS-based atomic min (device function)
#pragma omp declare target
template <typename T>
T atomic_min_cas(T* address, T val) {
  T ret = *address;
  while (val < ret) {
    T old = ret;
    if (__atomic_compare_exchange_n(address, &old, val, false,
                                    __ATOMIC_RELAXED, __ATOMIC_RELAXED)) break;
    ret = old;
  }
  return ret;
}

template <typename T>
T atomic_max_cas(T* address, T val) {
  T ret = *address;
  while (val > ret) {
    T old = ret;
    if (__atomic_compare_exchange_n(address, &old, val, false,
                                    __ATOMIC_RELAXED, __ATOMIC_RELAXED)) break;
    ret = old;
  }
  return ret;
}

template <typename T>
T atomic_add_cas(T* address, T val) {
  T old, newval, ret = *address;
  do {
    old    = ret;
    newval = old + val;
    if (__atomic_compare_exchange_n(address, &old, newval, false,
                                    __ATOMIC_RELAXED, __ATOMIC_RELAXED)) break;
    ret = old;
  } while (true);
  return ret;
}

double atomic_min_double(double* address, double val) {
  unsigned long long* ptr = reinterpret_cast<unsigned long long*>(address);
  unsigned long long val_bits;
  memcpy(&val_bits, &val, sizeof(double));
  unsigned long long ret = *ptr;
  double ret_d;
  memcpy(&ret_d, &ret, sizeof(double));
  while (val < ret_d) {
    unsigned long long old_bits = ret;
    if (__atomic_compare_exchange_n(ptr, &old_bits, val_bits, false,
                                    __ATOMIC_RELAXED, __ATOMIC_RELAXED)) break;
    ret = old_bits;
    memcpy(&ret_d, &ret, sizeof(double));
  }
  double result;
  memcpy(&result, &ret, sizeof(double));
  return result;
}

double atomic_max_double(double* address, double val) {
  unsigned long long* ptr = reinterpret_cast<unsigned long long*>(address);
  unsigned long long val_bits;
  memcpy(&val_bits, &val, sizeof(double));
  unsigned long long ret = *ptr;
  double ret_d;
  memcpy(&ret_d, &ret, sizeof(double));
  while (val > ret_d) {
    unsigned long long old_bits = ret;
    if (__atomic_compare_exchange_n(ptr, &old_bits, val_bits, false,
                                    __ATOMIC_RELAXED, __ATOMIC_RELAXED)) break;
    ret = old_bits;
    memcpy(&ret_d, &ret, sizeof(double));
  }
  double result;
  memcpy(&result, &ret, sizeof(double));
  return result;
}

double atomic_add_double(double* address, double val) {
  unsigned long long* ptr = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old_bits, new_bits;
  double old_d, new_d;
  old_bits = *ptr;
  do {
    memcpy(&old_d, &old_bits, sizeof(double));
    new_d = old_d + val;
    memcpy(&new_bits, &new_d, sizeof(double));
    if (__atomic_compare_exchange_n(ptr, &old_bits, new_bits, false,
                                    __ATOMIC_RELAXED, __ATOMIC_RELAXED)) break;
  } while (true);
  double result;
  memcpy(&result, &old_bits, sizeof(double));
  return result;
}
#pragma omp end declare target

template <typename T>
void testMin(T* h_ptr, const int repeat, const char* name) {
  T* d_val = (T*)malloc(sizeof(T));
  d_val[0] = *h_ptr;
  #pragma omp target enter data map(to: d_val[0:1])

  auto start = std::chrono::steady_clock::now();
  for (int n = 0; n < repeat; n++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int idx = 0; idx < NUM_BLOCKS * BLOCK_SIZE; idx++) {
      T i = (T)(idx + 1);
      atomic_min_cas(&d_val[0], i);
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Atomic min for data type %s | ", name);
  printf("Average execution time: %f (s)\n", (time * 1e-9f) / repeat);

  #pragma omp target update from(d_val[0:1])
  *h_ptr = d_val[0];
  #pragma omp target exit data map(delete: d_val[0:1])
  free(d_val);
}

template<>
void testMin<double>(double* h_ptr, const int repeat, const char* name) {
  double* d_val = (double*)malloc(sizeof(double));
  d_val[0] = *h_ptr;
  #pragma omp target enter data map(to: d_val[0:1])

  auto start = std::chrono::steady_clock::now();
  for (int n = 0; n < repeat; n++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int idx = 0; idx < NUM_BLOCKS * BLOCK_SIZE; idx++) {
      double i = (double)(idx + 1);
      atomic_min_double(&d_val[0], i);
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Atomic min for data type %s | ", name);
  printf("Average execution time: %f (s)\n", (time * 1e-9f) / repeat);

  #pragma omp target update from(d_val[0:1])
  *h_ptr = d_val[0];
  #pragma omp target exit data map(delete: d_val[0:1])
  free(d_val);
}

template <typename T>
void testMax(T* h_ptr, const int repeat, const char* name) {
  T* d_val = (T*)malloc(sizeof(T));
  d_val[0] = *h_ptr;
  #pragma omp target enter data map(to: d_val[0:1])

  auto start = std::chrono::steady_clock::now();
  for (int n = 0; n < repeat; n++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int idx = 0; idx < NUM_BLOCKS * BLOCK_SIZE; idx++) {
      T i = (T)(idx + 1);
      atomic_max_cas(&d_val[0], i);
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Atomic max for data type %s | ", name);
  printf("Average execution time: %f (s)\n", (time * 1e-9f) / repeat);

  #pragma omp target update from(d_val[0:1])
  *h_ptr = d_val[0];
  #pragma omp target exit data map(delete: d_val[0:1])
  free(d_val);
}

template<>
void testMax<double>(double* h_ptr, const int repeat, const char* name) {
  double* d_val = (double*)malloc(sizeof(double));
  d_val[0] = *h_ptr;
  #pragma omp target enter data map(to: d_val[0:1])

  auto start = std::chrono::steady_clock::now();
  for (int n = 0; n < repeat; n++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int idx = 0; idx < NUM_BLOCKS * BLOCK_SIZE; idx++) {
      double i = (double)(idx + 1);
      atomic_max_double(&d_val[0], i);
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Atomic max for data type %s | ", name);
  printf("Average execution time: %f (s)\n", (time * 1e-9f) / repeat);

  #pragma omp target update from(d_val[0:1])
  *h_ptr = d_val[0];
  #pragma omp target exit data map(delete: d_val[0:1])
  free(d_val);
}

template <typename T>
void testAdd(T* h_ptr, const int repeat, const char* name) {
  T* d_val = (T*)malloc(sizeof(T));
  d_val[0] = *h_ptr;
  #pragma omp target enter data map(to: d_val[0:1])

  auto start = std::chrono::steady_clock::now();
  for (int n = 0; n < repeat; n++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int idx = 0; idx < NUM_BLOCKS * BLOCK_SIZE; idx++) {
      T i = (T)(idx + 1);
      atomic_add_cas(&d_val[0], i);
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Atomic add for data type %s | ", name);
  printf("Average execution time: %f (s)\n", (time * 1e-9f) / repeat);

  #pragma omp target update from(d_val[0:1])
  *h_ptr = d_val[0];
  #pragma omp target exit data map(delete: d_val[0:1])
  free(d_val);
}

template<>
void testAdd<double>(double* h_ptr, const int repeat, const char* name) {
  double* d_val = (double*)malloc(sizeof(double));
  d_val[0] = *h_ptr;
  #pragma omp target enter data map(to: d_val[0:1])

  auto start = std::chrono::steady_clock::now();
  for (int n = 0; n < repeat; n++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int idx = 0; idx < NUM_BLOCKS * BLOCK_SIZE; idx++) {
      double i = (double)(idx + 1);
      atomic_add_double(&d_val[0], i);
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Atomic add for data type %s | ", name);
  printf("Average execution time: %f (s)\n", (time * 1e-9f) / repeat);

  #pragma omp target update from(d_val[0:1])
  *h_ptr = d_val[0];
  #pragma omp target exit data map(delete: d_val[0:1])
  free(d_val);
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  unsigned long long res_u64[3] = {ULONG_MAX, 0, 0};
  long long          res_s64[3] = {LONG_MAX, LONG_MIN, 0};
  double             res_f64[3] = {DBL_MAX, DBL_MIN, 0.0};

  testMin<unsigned long long>(res_u64,     repeat, "U64");
  testMin<long long>         (res_s64,     repeat, "S64");
  testMin<double>            (res_f64,     repeat, "F64");

  testMax<unsigned long long>(res_u64 + 1, repeat, "U64");
  testMax<long long>         (res_s64 + 1, repeat, "S64");
  testMax<double>            (res_f64 + 1, repeat, "F64");

  testAdd<unsigned long long>(res_u64 + 2, 1, "U64");
  testAdd<long long>         (res_s64 + 2, 1, "S64");
  testAdd<double>            (res_f64 + 2, 1, "F64");

  const unsigned long long bound = (unsigned long long)NUM_BLOCKS * BLOCK_SIZE;
  unsigned long long sum = 0;
  for (unsigned long long i = 1; i <= bound; i++) sum += i;

  bool error = false;
  if (res_u64[0] != 1ULL || res_s64[0] != 1LL || res_f64[0] != 1.0) {
    error = true;
    printf("atomic min results: %llu %lld %lf\n",
           res_u64[0], res_s64[0], res_f64[0]);
  }
  if (res_u64[1] != bound ||
      res_s64[1] != (long long)bound ||
      res_f64[1] != (double)bound) {
    error = true;
    printf("atomic max results: %llu %lld %lf\n",
           res_u64[1], res_s64[1], res_f64[1]);
  }
  if (res_u64[2] != sum ||
      res_s64[2] != (long long)sum ||
      res_f64[2] != (double)sum) {
    error = true;
    printf("atomic add results: %llu %lld %lf\n",
           res_u64[2], res_s64[2], res_f64[2]);
  }
  printf("%s\n", error ? "FAIL" : "PASS");
  return 0;
}
