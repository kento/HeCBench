#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>

// ── Portable bit-reinterpret ────────────────────────────────────────────────

template <typename To, typename From>
#pragma omp declare target
To bit_reinterpret(From f)
{
  static_assert(sizeof(To) == sizeof(From), "bit_reinterpret: size mismatch");
  union { From from; To to; } u;
  u.from = f;
  return u.to;
}
#pragma omp end declare target

#pragma omp declare target
int double2hiint(double x)
{
  return (int)(bit_reinterpret<long long>(x) >> 32);
}

int double2loint(double x)
{
  return (int)(bit_reinterpret<long long>(x) & 0xFFFFFFFFLL);
}

double hiloint2double(int hi, int lo)
{
  long long v = ((long long)(unsigned int)hi << 32) | (unsigned int)lo;
  return bit_reinterpret<double>(v);
}

long long cast1(double x)
{
  int           r1 = 0;
  unsigned int  r2 = 0;
  long long     r3 = 0;
  unsigned long long r4 = 0;

  r1 ^= double2hiint(x);
  r1 ^= double2loint(x);

  r1 ^= (int)floor(x);
  r1 ^= (int)round(x);
  r1 ^= (int)ceil(x);
  r1 ^= (int)x;

  r1 ^= (int)floorf((float)x);
  r1 ^= (int)roundf((float)x);
  r1 ^= (int)ceilf((float)x);
  r1 ^= (int)(float)x;

  r1 ^= bit_reinterpret<int>((float)x);

  r2 ^= (unsigned int)(unsigned long long)floor(x);
  r2 ^= (unsigned int)(unsigned long long)round(x);
  r2 ^= (unsigned int)(unsigned long long)ceil(x);
  r2 ^= (unsigned int)(unsigned long long)x;

  r2 ^= (unsigned int)(unsigned long long)floorf((float)x);
  r2 ^= (unsigned int)(unsigned long long)roundf((float)x);
  r2 ^= (unsigned int)(unsigned long long)ceilf((float)x);
  r2 ^= (unsigned int)(unsigned long long)(float)x;

  r2 ^= bit_reinterpret<unsigned int>((float)x);

  r3 ^= (long long)floor(x);
  r3 ^= (long long)round(x);
  r3 ^= (long long)ceil(x);
  r3 ^= (long long)x;

  r3 ^= (long long)floorf((float)x);
  r3 ^= (long long)roundf((float)x);
  r3 ^= (long long)ceilf((float)x);
  r3 ^= (long long)(float)x;

  r3 ^= bit_reinterpret<long long>(x);

  r4 ^= (unsigned long long)floor(x);
  r4 ^= (unsigned long long)round(x);
  r4 ^= (unsigned long long)ceil(x);
  r4 ^= (unsigned long long)x;

  r4 ^= (unsigned long long)floorf((float)x);
  r4 ^= (unsigned long long)roundf((float)x);
  r4 ^= (unsigned long long)ceilf((float)x);
  r4 ^= (unsigned long long)(float)x;

  return (long long)(
    (unsigned long long)((unsigned int)r1 + r2) +
    (unsigned long long)r3 + r4);
}

long long cast2(long long x)
{
  float  r1 = 0.f;
  double r2 = 0.0;

  r1 += (float)hiloint2double((int)(x >> 32), (int)x);

  r1 += (float)(int)x;
  r1 += (float)(int)x;
  r1 += (float)(int)x;
  r1 += (float)(int)x;

  r1 += (float)(unsigned int)x;
  r1 += (float)(unsigned int)x;
  r1 += (float)(unsigned int)x;
  r1 += (float)(unsigned int)x;

  r1 += bit_reinterpret<float>((int)x);
  r1 += bit_reinterpret<float>((unsigned int)x);

  r1 += (float)(long long)x;
  r1 += (float)(long long)x;
  r1 += (float)(long long)x;
  r1 += (float)(long long)x;

  r1 += (float)(unsigned long long)x;
  r1 += (float)(unsigned long long)x;
  r1 += (float)(unsigned long long)x;
  r1 += (float)(unsigned long long)x;

  r2 += (double)(int)x;
  r2 += (double)(unsigned int)x;

  r2 += (double)(long long)x;
  r2 += (double)(long long)x;
  r2 += (double)(long long)x;
  r2 += (double)(long long)x;

  r2 += (double)(unsigned long long)x;
  r2 += (double)(unsigned long long)x;
  r2 += (double)(unsigned long long)x;
  r2 += (double)(unsigned long long)x;

  r2 += bit_reinterpret<double>(x);

  return bit_reinterpret<long long>((double)r1 + r2);
}
#pragma omp end declare target

int main(int argc, char *argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int n      = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  double    *h_in1  = new double[n];
  long long *h_in2  = new long long[n];

  for (int i = 1; i <= n; i++) {
    h_in1[i - 1] = 22.44 / i;
    h_in2[i - 1] = (long long)0x403670A3D70A3D71LL;
  }

  double*    d_in1  = (double*)malloc(n * sizeof(double));
  long long* d_out1 = (long long*)malloc(n * sizeof(long long));
  long long* d_in2  = (long long*)malloc(n * sizeof(long long));
  long long* d_out2 = (long long*)malloc(n * sizeof(long long));

  for (int i = 0; i < n; i++) { d_in1[i] = h_in1[i]; d_in2[i] = h_in2[i]; }

  #pragma omp target enter data map(to: d_in1[0:n], d_in2[0:n]) \
      map(alloc: d_out1[0:n], d_out2[0:n])

  // cast1 benchmark
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < n; i++) {
      d_out1[i] = cast1(d_in1[i]);
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  printf("Average execution time of the cast intrinsics kernel (from FP): %f (us)\n",
         (std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
          * 1e-3f) / repeat);

  #pragma omp target update from(d_out1[0:n])
  long long checksum = 0;
  for (int i = 0; i < n; i++) checksum ^= d_out1[i];
  printf("Checksum = %llx\n", checksum);

  // cast2 benchmark
  t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < n; i++) {
      d_out2[i] = cast2(d_in2[i]);
    }
  }
  t1 = std::chrono::steady_clock::now();
  printf("Average execution time of the cast intrinsics kernel (to FP): %f (us)\n",
         (std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
          * 1e-3f) / repeat);

  #pragma omp target update from(d_out2[0:n])
  checksum = 0;
  for (int i = 0; i < n; i++) checksum ^= d_out2[i];
  printf("Checksum = %llx\n", checksum);

  #pragma omp target exit data map(delete: d_in1[0:n], d_in2[0:n], d_out1[0:n], d_out2[0:n])
  free(d_in1); free(d_out1); free(d_in2); free(d_out2);
  delete[] h_in1;
  delete[] h_in2;
  return 0;
}
