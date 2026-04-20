// OpenMP target offload port of merkle-kokkos benchmark.
// Merkle tree using Rescue Prime hash over F(2^64 - 2^32 + 1)
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <random>
#include <cstdint>
#include <climits>
#include <cstring>
#include <cstdio>
#include <vector>

typedef unsigned long ulong;

// Field modulus p = 2^64 - 2^32 + 1
constexpr uint64_t MOD        = ((((uint64_t)1 << 63) - ((uint64_t)1 << 31)) << 1) + 1;
constexpr int      RATE_WIDTH  = 8;
constexpr int      DIGEST_SIZE = 4;
constexpr int      STATE_WIDTH = 12;
constexpr int      NUM_ROUNDS  = 7;

// MDS matrix: 12x12 field elements stored row-major
static const uint64_t HOST_MDS[144] = {
  7,  23, 8,  26, 20, 11, 3,  7,  4,  23, 8,  26,
  23, 8,  26, 20, 11, 3,  7,  4,  23, 8,  26, 20,
  8,  26, 20, 11, 3,  7,  4,  23, 8,  26, 20, 11,
  26, 20, 11, 3,  7,  4,  23, 8,  26, 20, 11, 3,
  20, 11, 3,  7,  4,  23, 8,  26, 20, 11, 3,  7,
  11, 3,  7,  4,  23, 8,  26, 20, 11, 3,  7,  4,
  3,  7,  4,  23, 8,  26, 20, 11, 3,  7,  4,  23,
  7,  4,  23, 8,  26, 20, 11, 3,  7,  4,  23, 8,
  4,  23, 8,  26, 20, 11, 3,  7,  4,  23, 8,  26,
  23, 8,  26, 20, 11, 3,  7,  4,  23, 8,  26, 20,
  8,  26, 20, 11, 3,  7,  4,  23, 8,  26, 20, 11,
  26, 20, 11, 3,  7,  4,  23, 8,  26, 20, 11, 3,
};

// Round constants: 2*NUM_ROUNDS rows x 12 columns
static const uint64_t HOST_RC[168] = {
  13,  4,  5, 17,  8, 15,  3, 19,  6, 10,  2, 14,
  11, 16,  7,  9, 12,  1, 18,  0, 13,  4,  5, 17,
   8, 15,  3, 19,  6, 10,  2, 14, 11, 16,  7,  9,
  12,  1, 18,  0, 13,  4,  5, 17,  8, 15,  3, 19,
   6, 10,  2, 14, 11, 16,  7,  9, 12,  1, 18,  0,
  13,  4,  5, 17,  8, 15,  3, 19,  6, 10,  2, 14,
  11, 16,  7,  9, 12,  1, 18,  0, 13,  4,  5, 17,
   8, 15,  3, 19,  6, 10,  2, 14, 11, 16,  7,  9,
  12,  1, 18,  0, 13,  4,  5, 17,  8, 15,  3, 19,
   6, 10,  2, 14, 11, 16,  7,  9, 12,  1, 18,  0,
  13,  4,  5, 17,  8, 15,  3, 19,  6, 10,  2, 14,
  11, 16,  7,  9, 12,  1, 18,  0, 13,  4,  5, 17,
   8, 15,  3, 19,  6, 10,  2, 14, 11, 16,  7,  9,
  12,  1, 18,  0, 13,  4,  5, 17,  8, 15,  3, 19,
};

// ---------------------------------------------------------------------------
// Device functions: wrapped in declare target region
// ---------------------------------------------------------------------------
#pragma omp declare target

struct ulong4 {
  uint64_t x, y, z, w;
};

inline ulong4 make_ulong4(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
  return {a, b, c, d};
}

inline uint64_t umul64hi(uint64_t a, uint64_t b) {
#if defined(__SIZEOF_INT128__)
  return (uint64_t)(((unsigned __int128)a * b) >> 64);
#else
  uint64_t lo_a = a & 0xFFFFFFFFULL, hi_a = a >> 32;
  uint64_t lo_b = b & 0xFFFFFFFFULL, hi_b = b >> 32;
  uint64_t mid1 = lo_a * hi_b, mid2 = hi_a * lo_b;
  uint64_t lo   = lo_a * lo_b;
  uint64_t carry = ((lo >> 32) + (mid1 & 0xFFFFFFFFULL) + (mid2 & 0xFFFFFFFFULL)) >> 32;
  return hi_a * hi_b + (mid1 >> 32) + (mid2 >> 32) + carry;
#endif
}

inline ulong4 operator*(const ulong4& a, const ulong4& b) { return {a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w}; }
inline ulong4 operator+(const ulong4& a, const ulong4& b) { return {a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w}; }
inline ulong4 operator-(const ulong4& a, const ulong4& b) { return {a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w}; }
inline ulong4 operator-(uint64_t a,       const ulong4& b) { return {a-b.x, a-b.y, a-b.z, a-b.w}; }
inline ulong4 operator&(const ulong4& a, uint64_t b)       { return {a.x&b, a.y&b, a.z&b, a.w&b}; }
inline ulong4 operator>>(const ulong4& a, int b)           { return {a.x>>b, a.y>>b, a.z>>b, a.w>>b}; }
inline ulong4 operator<<(const ulong4& a, int b)           { return {a.x<<b, a.y<<b, a.z<<b, a.w<<b}; }

inline ulong4 cmp_lt(const ulong4& a, const ulong4& b) {
  return {(a.x<b.x)?ULONG_MAX:0, (a.y<b.y)?ULONG_MAX:0,
          (a.z<b.z)?ULONG_MAX:0, (a.w<b.w)?ULONG_MAX:0};
}
inline ulong4 cmp_gt(const ulong4& a, const ulong4& b) {
  return {(a.x>b.x)?ULONG_MAX:0, (a.y>b.y)?ULONG_MAX:0,
          (a.z>b.z)?ULONG_MAX:0, (a.w>b.w)?ULONG_MAX:0};
}
inline ulong4 cmp_ge(const ulong4& a, const ulong4& b) {
  return {(a.x>=b.x)?ULONG_MAX:0, (a.y>=b.y)?ULONG_MAX:0,
          (a.z>=b.z)?ULONG_MAX:0, (a.w>=b.w)?ULONG_MAX:0};
}
inline ulong4 mul_hi(const ulong4& a, const ulong4& b) {
  return {umul64hi(a.x,b.x), umul64hi(a.y,b.y), umul64hi(a.z,b.z), umul64hi(a.w,b.w)};
}

inline uint64_t ff_p_add(uint64_t a, uint64_t b) {
  if (b >= MOD) b -= MOD;
  uint64_t res = a + b;
  bool ov = (a > UINT64_MAX - b);
  uint64_t t = (uint64_t)(-(uint32_t)(ov ? 1 : 0));
  res += t;
  bool ov2 = (res < t && ov);
  res += (uint64_t)(-(uint32_t)(ov2 ? 1 : 0));
  return res;
}

inline uint64_t ff_p_sub(uint64_t a, uint64_t b) {
  if (b >= MOD) b -= MOD;
  uint64_t res = a - b;
  bool un = (a < b);
  uint64_t t = (uint64_t)(-(uint32_t)(un ? 1 : 0));
  res -= t;
  bool un2 = (res > a && un);
  res += (uint64_t)(-(uint32_t)(un2 ? 1 : 0));
  return res;
}

inline uint64_t ff_p_mult(uint64_t a, uint64_t b) {
  if (b >= MOD) b -= MOD;
  uint64_t ab    = a * b;
  uint64_t cd_hi = umul64hi(a, b);
  uint64_t c     = cd_hi & 0x00000000ffffffffULL;
  uint64_t d     = cd_hi >> 32;
  uint64_t res   = ab - d;
  bool un = (ab < d);
  uint64_t t = (uint64_t)(-(uint32_t)(un ? 1 : 0));
  res -= t;
  uint64_t t1   = (c << 32) - c;
  uint64_t res2 = res + t1;
  bool ov = (res > UINT64_MAX - t1);
  res2 += (uint64_t)(-(uint32_t)(ov ? 1 : 0));
  return res2;
}

inline uint64_t ff_p_pow(uint64_t a, uint64_t b) {
  if (b == 0) return 1;
  if (b == 1) return a;
  if (a == 0) return 0;
  int bits = 64;
  while (bits > 0 && !((b >> (bits-1)) & 1)) bits--;
  uint64_t r = (b & 1) ? a : 1;
  for (int i = 1; i < bits; i++) {
    a = ff_p_mult(a, a);
    if ((b >> i) & 1) r = ff_p_mult(r, a);
  }
  return r;
}

inline uint64_t ff_p_inv(uint64_t a) {
  if (a >= MOD) a -= MOD;
  if (a == 0) return 0;
  return ff_p_pow(a, MOD - 2);
}

inline ulong4 ff_p_vec_mul_(ulong4 a, ulong4 b) {
  if (b.x >= MOD) b.x -= MOD;
  if (b.y >= MOD) b.y -= MOD;
  if (b.z >= MOD) b.z -= MOD;
  if (b.w >= MOD) b.w -= MOD;
  ulong4 ab  = a * b;
  ulong4 cd  = mul_hi(a, b);
  ulong4 c   = cd & 0x00000000ffffffffULL;
  ulong4 d   = cd >> 32;
  ulong4 res0 = ab - d;
  ulong4 un0  = cmp_lt(ab, d);
  res0 = res0 - (un0 & 1ULL);
  ulong4 t1   = (c << 32) - c;
  ulong4 res1 = res0 + t1;
  ulong4 ov0  = cmp_gt(res0, make_ulong4(UINT64_MAX,UINT64_MAX,UINT64_MAX,UINT64_MAX) - t1);
  res1 = res1 + (ov0 & 1ULL);
  return res1;
}

inline void ff_p_vec_mul(const ulong4* a, const ulong4* b, ulong4* c) {
  c[0] = ff_p_vec_mul_(a[0], b[0]);
  c[1] = ff_p_vec_mul_(a[1], b[1]);
  c[2] = ff_p_vec_mul_(a[2], b[2]);
}

inline ulong4 ff_p_vec_add_(const ulong4& a, const ulong4& b) {
  ulong4 res;
  res.x = ff_p_add(a.x, b.x);
  res.y = ff_p_add(a.y, b.y);
  res.z = ff_p_add(a.z, b.z);
  res.w = ff_p_add(a.w, b.w);
  return res;
}

inline void ff_p_vec_add(const ulong4* a, const ulong4* b, ulong4* c) {
  c[0] = ff_p_vec_add_(a[0], b[0]);
  c[1] = ff_p_vec_add_(a[1], b[1]);
  c[2] = ff_p_vec_add_(a[2], b[2]);
}

// Rescue Prime hash: input[0:8] -> digest[0:4]
inline void rescue_prime_hash(
    const uint64_t* input,   // RATE_WIDTH = 8 elements
    uint64_t*       digest,  // DIGEST_SIZE = 4 elements
    const uint64_t* mds,     // 144 elements (12x12)
    const uint64_t* rc)      // 168 elements (14x12)
{
  // exponent for 7th root in GF(p): modular_inverse(7, p-1)
  const uint64_t inv7_exp = 10540996611094048183ULL;

  uint64_t state[12] = {};
  for (int i = 0; i < 8; i++) state[i] = input[i];

  for (int r = 0; r < 7; r++) {
    // Add forward round constants
    for (int i = 0; i < 12; i++)
      state[i] = ff_p_add(state[i], rc[r*24 + i]);
    // S-box: x^7
    for (int i = 0; i < 12; i++)
      state[i] = ff_p_pow(state[i], 7);
    // MDS multiply
    uint64_t tmp[12] = {};
    for (int i = 0; i < 12; i++)
      for (int j = 0; j < 12; j++)
        tmp[i] = ff_p_add(tmp[i], ff_p_mult(mds[i*12+j], state[j]));
    for (int i = 0; i < 12; i++) state[i] = tmp[i];
    // Add backward round constants
    for (int i = 0; i < 12; i++)
      state[i] = ff_p_add(state[i], rc[r*24 + 12 + i]);
    // Inverse S-box: x^(1/7)
    for (int i = 0; i < 12; i++)
      state[i] = ff_p_pow(state[i], inv7_exp);
    // MDS multiply
    for (int i = 0; i < 12; i++) tmp[i] = 0;
    for (int i = 0; i < 12; i++)
      for (int j = 0; j < 12; j++)
        tmp[i] = ff_p_add(tmp[i], ff_p_mult(mds[i*12+j], state[j]));
    for (int i = 0; i < 12; i++) state[i] = tmp[i];
  }
  for (int i = 0; i < 4; i++) digest[i] = state[i];
}

#pragma omp end declare target

// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
  const int N    = 1 << 14; // 16384 leaves (power of 2)
  const int nrep = 3;

  printf("Merkle tree (Rescue Prime hash, n=%d leaves)\n", N);

  // total_nodes = 2*N: leaves at [N..2N-1], internal + root at [1..N-1], [0] unused
  const int total_nodes = N * 2;
  const int nodes_sz    = total_nodes * DIGEST_SIZE;
  const int input_sz    = N * RATE_WIDTH;

  // Host data
  std::vector<uint64_t> h_input(input_sz);
  std::mt19937_64 rng(42);
  for (int i = 0; i < input_sz; i++) h_input[i] = rng() % MOD;

  // Allocate device-mapped arrays
  uint64_t* d_nodes = (uint64_t*)malloc(nodes_sz  * sizeof(uint64_t));
  uint64_t* d_input = (uint64_t*)malloc(input_sz  * sizeof(uint64_t));
  uint64_t* d_mds   = (uint64_t*)malloc(144        * sizeof(uint64_t));
  uint64_t* d_rc    = (uint64_t*)malloc(168        * sizeof(uint64_t));

  // Populate host arrays before uploading
  memcpy(d_input, h_input.data(), input_sz * sizeof(uint64_t));
  memcpy(d_mds,   HOST_MDS,       144       * sizeof(uint64_t));
  memcpy(d_rc,    HOST_RC,        168       * sizeof(uint64_t));

  // Map arrays to device
  #pragma omp target enter data map(alloc: d_nodes[0:nodes_sz])
  #pragma omp target enter data map(to: d_input[0:input_sz])
  #pragma omp target enter data map(to: d_mds[0:144])
  #pragma omp target enter data map(to: d_rc[0:168])

  double total_t = 0.0;

  for (int rep = 0; rep < nrep; rep++) {
    // Zero out node array on device
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < nodes_sz; i++) d_nodes[i] = 0ULL;

    double t0 = omp_get_wtime();

    // Step 1: Hash leaves -> d_nodes[(N+i)*DIGEST_SIZE]
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < N; i++) {
      const uint64_t* inp = &d_input[i * RATE_WIDTH];
      uint64_t*       out = &d_nodes[(N + i) * DIGEST_SIZE];
      rescue_prime_hash(inp, out, d_mds, d_rc);
    }

    // Step 2: Build tree bottom-up, one kernel per level
    for (int level_sz = N/2; level_sz >= 1; level_sz /= 2) {
      const int offset = level_sz; // parent nodes start at [offset..offset+level_sz-1]
      #pragma omp target teams distribute parallel for thread_limit(256)
      for (int i = 0; i < level_sz; i++) {
        const int left  = 2 * (offset + i);
        const int right = left + 1;
        uint64_t inp[8];
        for (int k = 0; k < 4; k++) {
          inp[k]   = d_nodes[left  * DIGEST_SIZE + k];
          inp[k+4] = d_nodes[right * DIGEST_SIZE + k];
        }
        uint64_t* out = &d_nodes[(offset + i) * DIGEST_SIZE];
        rescue_prime_hash(inp, out, d_mds, d_rc);
      }
    }

    double t1 = omp_get_wtime();
    total_t += t1 - t0;
  }

  printf("Average time: %.6f s\n", total_t / nrep);

  // Copy root hash back to host (root is at node index 1)
  #pragma omp target update from(d_nodes[0:nodes_sz])
  std::cout << "Root hash: ";
  for (int i = 0; i < DIGEST_SIZE; i++)
    std::cout << std::hex << d_nodes[DIGEST_SIZE + i] << " ";
  std::cout << std::dec << "\n";

  // Cleanup
  #pragma omp target exit data map(delete: d_nodes[0:nodes_sz])
  #pragma omp target exit data map(delete: d_input[0:input_sz])
  #pragma omp target exit data map(delete: d_mds[0:144])
  #pragma omp target exit data map(delete: d_rc[0:168])

  free(d_nodes);
  free(d_input);
  free(d_mds);
  free(d_rc);

  return 0;
}
