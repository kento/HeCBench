/*
 * OpenMP target offloading port of bitpermute (NTT bit-reversal permutation).
 */

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
#include <omp.h>

#define MAX_LG_DOMAIN_SIZE 28
typedef long fr_t;

#if MAX_LG_DOMAIN_SIZE <= 32
typedef unsigned int index_t;
#else
typedef size_t index_t;
#endif

#pragma omp declare target
unsigned int brev32(unsigned int x) {
  x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
  x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
  x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
  x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
  x = (x >> 16) | (x << 16);
  return x;
}

unsigned long long brev64(unsigned long long x) {
  return ((unsigned long long)brev32((unsigned int)x) << 32)
       | (unsigned long long)brev32((unsigned int)(x >> 32));
}

index_t bit_rev(index_t i, unsigned int nbits) {
  if (sizeof(i) == 4 || nbits <= 32)
    return (index_t)(brev32((unsigned int)i) >> (32 - nbits));
  else
    return (index_t)(brev64((unsigned long long)i) >> (64 - nbits));
}
#pragma omp end declare target

template<typename T>
T reverseBits(T num) {
  unsigned int NO_OF_BITS = sizeof(T) * 8;
  T reverse_num = 0;
  for (unsigned int i = 0; i < NO_OF_BITS; i++)
    if (num & ((T)1 << i))
      reverse_num |= (T)1 << ((NO_OF_BITS - 1) - i);
  return reverse_num;
}

template<typename T>
T rev_cpu(T i, unsigned int nbits) {
  if (sizeof(T) == 4 || nbits <= 32)
    return reverseBits(i) >> (8*sizeof(unsigned int) - nbits);
  else
    return reverseBits(i) >> (8*sizeof(unsigned long long) - nbits);
}

void bit_rev_cpu(fr_t* out, const fr_t* in, uint32_t lg_domain_size) {
  uint32_t domain_size = 1 << lg_domain_size;
  for (uint32_t i = 0; i < domain_size; i++) {
    index_t r = rev_cpu(i, lg_domain_size);
    if (i < r) {
      out[r] = in[i];
      out[i] = in[r];
    }
  }
}

void bit_permute(const int lg_domain_size, const int repeat) {
  assert(lg_domain_size <= MAX_LG_DOMAIN_SIZE);
  size_t domain_size = (size_t)1 << lg_domain_size;
  printf("Domain size is %zu\n", domain_size);

  fr_t* d_inout = (fr_t*)malloc(domain_size * sizeof(fr_t));
  std::vector<fr_t> h_inout(domain_size), h_out(domain_size);
  for (size_t i = 0; i < domain_size; i++) h_inout[i] = h_out[i] = (fr_t)i;
  memcpy(d_inout, h_inout.data(), domain_size * sizeof(fr_t));

  #pragma omp target enter data map(to: d_inout[0:domain_size])

  // GPU bit reversal (warmup + verify)
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (size_t i = 0; i < domain_size; i++) {
    index_t r = bit_rev((index_t)i, lg_domain_size);
    if (i < (size_t)r) {
      fr_t t0 = d_inout[i];
      fr_t t1 = d_inout[r];
      d_inout[i] = t1;
      d_inout[r] = t0;
    }
  }

  #pragma omp target update from(d_inout[0:domain_size])

  // CPU reference
  bit_rev_cpu(h_out.data(), h_inout.data(), lg_domain_size);

  int error = memcmp(h_out.data(), d_inout, domain_size * sizeof(fr_t));
  printf("%s\n", error ? "FAIL" : "PASS");

  // Reset for timing
  memcpy(d_inout, h_inout.data(), domain_size * sizeof(fr_t));
  #pragma omp target update to(d_inout[0:domain_size])

  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (size_t i = 0; i < domain_size; i++) {
      index_t rev = bit_rev((index_t)i, lg_domain_size);
      if (i < (size_t)rev) {
        fr_t t0 = d_inout[i];
        fr_t t1 = d_inout[rev];
        d_inout[i] = t1;
        d_inout[rev] = t0;
      }
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the kernel: %f (us)\n\n", (time * 1e-3f) / repeat);

  #pragma omp target exit data map(delete: d_inout[0:domain_size])
  free(d_inout);
}

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  bit_permute(10, repeat);
  bit_permute(11, repeat);
  bit_permute(15, repeat);
  bit_permute(27, repeat);
  bit_permute(28, repeat);
  return 0;
}
