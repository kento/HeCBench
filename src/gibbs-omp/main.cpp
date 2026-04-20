/*---------------------------------------------------------------
  OpenMP target offloading port of gibbs-kokkos (Gibbs sampler)
----------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>

#define PI 3.14159265359f

// Host-side random helpers
static float rnorm()
{
  float U1 = rand() / float(RAND_MAX);
  float U2 = rand() / float(RAND_MAX);
  return sqrtf(-2.f * logf(U1)) * cosf(2.f * PI * U2);
}

static float rgamma_host(float a, float b)
{
  if (a > 1.f) {
    float d = a - 1.f / 3.f;
    float c = 1.f / sqrtf(9.f * d);
    for (;;) {
      float Z = rnorm();
      if (Z > -1.f / c) {
        float V = powf(1.f + c * Z, 3.f);
        float U = rand() / (float)RAND_MAX;
        if (logf(U) <= 0.5f * Z * Z + d - d * V + d * logf(V))
          return d * V / b;
      }
    }
  } else {
    float x = rgamma_host(a + 1.f, b);
    return x * powf(rand() / (float)RAND_MAX, 1.f / a);
  }
}

static float sample_a(float a, float b, int N, float log_sum)
{
  static float sigma = 2.f;
  float proposal = rnorm() * sigma + a;
  if (proposal <= 0.f) return a;
  float log_ar = (proposal - a) * log_sum
               + N * (proposal - a) * logf(b)
               - N * (lgammaf(proposal) - lgammaf(a));
  float U = rand() / float(RAND_MAX);
  if (logf(U) < log_ar) { sigma *= 1.1f; return proposal; }
  else                   { sigma /= 1.1f; return a; }
}

static float sample_b(float a, int N, float flat_sum)
{
  return rgamma_host(N * a + 1.f, flat_sum);
}

// Device-side LCG helpers
#pragma omp declare target
static inline float lcg_rand(unsigned int &s)
{
  s = 1664525u * s + 1013904223u;
  return (float)(s >> 8) * (1.0f / (float)(1u << 24));
}

static inline float lcg_normal(unsigned int &s)
{
  float u1, u2;
  do { u1 = lcg_rand(s); } while (u1 == 0.f);
  u2 = lcg_rand(s);
  return sqrtf(-2.f * logf(u1)) * cosf(2.f * 3.14159265359f * u2);
}

static inline float rgamma_device(unsigned int &s, float a, float b)
{
  float d = a - 1.f / 3.f;
  float c = 1.f / sqrtf(9.f * d);
  for (;;) {
    float Z = lcg_normal(s);
    if (Z > -1.f / c) {
      float V = powf(1.f + c * Z, 3.f);
      float U = lcg_rand(s);
      if (logf(U) <= 0.5f * Z * Z + d - d * V + d * logf(V))
        return d * V / b;
    }
  }
}
#pragma omp end declare target

int main(int argc, char *argv[])
{
  const int seed = 123;
  srand(seed);

  int trials = 1000;
  if (argc > 2) trials = atoi(argv[2]);

  if (argc < 2) { printf("Please provide input filename\n"); return 1; }
  FILE *fp = fopen(argv[1], "r");
  if (!fp) { printf("Cannot read file\n"); return 1; }

  int N = 0;
  char line[128];
  while (fgets(line, sizeof(line), fp)) N++;
  rewind(fp);

  int   *h_y = (int *)  malloc(sizeof(int)   * N);
  float *h_n = (float *)malloc(sizeof(float) * N);
  for (int i = 0; i < N; i++) fscanf(fp, "%d %f", &h_y[i], &h_n[i]);
  fclose(fp);

  int*          d_y      = (int*)          malloc(N * sizeof(int));
  float*        d_n      = (float*)        malloc(N * sizeof(float));
  float*        d_theta  = (float*)        malloc(N * sizeof(float));
  unsigned int* d_states = (unsigned int*) malloc(N * sizeof(unsigned int));

  for (int i = 0; i < N; i++) {
    d_y[i] = h_y[i];
    d_n[i] = h_n[i];
    d_states[i] = (unsigned int)(seed * 1664525u + (unsigned int)i * 22695477u + 1u);
  }

  #pragma omp target enter data map(to: d_y[0:N], d_n[0:N], d_states[0:N]) \
                                map(alloc: d_theta[0:N])

  float  a = 20.f, b = 1.f;
  double mean_a = 0.0, mean_b = 0.0, total_time = 0.0;

  for (int iter = 0; iter < trials; iter++) {
    auto t0 = std::chrono::steady_clock::now();

    const float fa = a, fb = b;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int id = 0; id < N; id++) {
      unsigned int s = d_states[id];
      const float hyperA = fa + (float)d_y[id];
      const float hyperB = fb + d_n[id];
      if (hyperA < 1.f)
        d_theta[id] = rgamma_device(s, hyperA + 1.f, hyperB)
                      * powf(lcg_rand(s), 1.f / hyperA);
      else
        d_theta[id] = rgamma_device(s, hyperA, hyperB);
      d_states[id] = s;
    }

    float flat_sum = 0.f, log_sum = 0.f;
    #pragma omp target teams distribute parallel for reduction(+:flat_sum) thread_limit(256)
    for (int id = 0; id < N; id++) flat_sum += d_theta[id];

    #pragma omp target teams distribute parallel for reduction(+:log_sum) thread_limit(256)
    for (int id = 0; id < N; id++) log_sum += logf(d_theta[id]);

    auto t1 = std::chrono::steady_clock::now();
    total_time += (double)std::chrono::duration_cast<
                    std::chrono::nanoseconds>(t1 - t0).count();

    a = sample_a(a, b, N, log_sum);
    mean_a += a;
    b = sample_b(a, N, flat_sum);
    mean_b += b;
  }

  printf("Average execution time of kernels: %f (us)\n",
         (total_time * 1e-3) / trials);
  printf("a = %lf (avg), b = %lf (avg)\n",
         mean_a / trials, mean_b / trials);

  #pragma omp target exit data map(delete: d_y[0:N], d_n[0:N], d_states[0:N], d_theta[0:N])
  free(d_y); free(d_n); free(d_theta); free(d_states);
  free(h_y); free(h_n);
  return 0;
}
