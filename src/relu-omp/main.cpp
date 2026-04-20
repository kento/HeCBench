// OpenMP target offloading port of relu benchmark.
// ReluGrad: backprop[i] = (feature[i] > 0) ? gradient[i] : 0
// Relu: clamp each signed byte in a packed int32 to >= 0.

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void ReluGrad_reference(int count,
                                const float *gradient,
                                const float *feature,
                                float       *backprop)
{
  for (int i = 0; i < count; i++)
    backprop[i] = (feature[i] > 0.f) ? gradient[i] : 0.f;
}

static void Relu_reference(int count, const int *input, int *output)
{
  for (int i = 0; i < count; i++) {
    signed char c1 = (signed char)( input[i]        & 0xFF);
    signed char c2 = (signed char)((input[i] >>  8) & 0xFF);
    signed char c3 = (signed char)((input[i] >> 16) & 0xFF);
    signed char c4 = (signed char)((input[i] >> 24) & 0xFF);
    unsigned x = (unsigned)(c1 > 0 ? c1 : 0);
    unsigned y = (unsigned)(c2 > 0 ? c2 : 0);
    unsigned z = (unsigned)(c3 > 0 ? c3 : 0);
    unsigned w = (unsigned)(c4 > 0 ? c4 : 0);
    output[i] = (int)(w << 24 | z << 16 | y << 8 | x);
  }
}

int main(int argc, char *argv[])
{
  if (argc != 3) {
    printf("Usage: %s <count> <repeat>\n", argv[0]);
    return 1;
  }
  const int count  = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  // =========================================================================
  // ReluGrad (float)
  // =========================================================================
  std::vector<float> h_gradient(count);
  std::vector<float> h_feature(count);
  std::vector<float> h_backprop(count);
  std::vector<float> r_backprop(count);

  std::mt19937 engine(19937);
  std::uniform_real_distribution<float> real_dist(-1.f, 1.f);
  for (int i = 0; i < count; i++) {
    h_feature[i]  = real_dist(engine);
    h_gradient[i] = 1.f;
  }
  ReluGrad_reference(count, h_gradient.data(), h_feature.data(), r_backprop.data());

  float *d_gradient = new float[count];
  float *d_feature  = new float[count];
  float *d_backprop = new float[count];
  memcpy(d_gradient, h_gradient.data(), count * sizeof(float));
  memcpy(d_feature,  h_feature.data(),  count * sizeof(float));

#pragma omp target enter data map(to: d_gradient[0:count], d_feature[0:count]) \
                              map(alloc: d_backprop[0:count])

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
#pragma omp target teams distribute parallel for thread_limit(256)
    for (int idx = 0; idx < count; idx++)
      d_backprop[idx] = (d_feature[idx] > 0.f) ? d_gradient[idx] : 0.f;
  }
  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of ReluGrad_impl1 Kernel: %f (us)\n", (time * 1e-3f) / repeat);

#pragma omp target update from(d_backprop[0:count])
  {
    int fail = 0;
    for (int i = 0; i < count; i++)
      if (std::fabs(d_backprop[i] - r_backprop[i]) > 1e-3f) { fail = 1; break; }
    printf("%s\n", fail ? "FAIL" : "PASS");
  }

  // impl2 (same kernel, separate timing)
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
#pragma omp target teams distribute parallel for thread_limit(256)
    for (int idx = 0; idx < count; idx++)
      d_backprop[idx] = (d_feature[idx] > 0.f) ? d_gradient[idx] : 0.f;
  }
  end  = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of ReluGrad_impl2 Kernel: %f (us)\n", (time * 1e-3f) / repeat);

#pragma omp target update from(d_backprop[0:count])
  {
    int fail = 0;
    for (int i = 0; i < count; i++)
      if (std::fabs(d_backprop[i] - r_backprop[i]) > 1e-3f) { fail = 1; break; }
    printf("%s\n", fail ? "FAIL" : "PASS");
  }

#pragma omp target exit data map(delete: d_gradient[0:count], d_feature[0:count], d_backprop[0:count])
  delete[] d_gradient; delete[] d_feature; delete[] d_backprop;

  // =========================================================================
  // Relu (packed int32 -> 4 x int8 clamped)
  // =========================================================================
  std::vector<int> h_in(count);
  std::vector<int> h_out(count);
  std::vector<int> r_out(count);

  std::uniform_int_distribution<unsigned char> int_dist(0, 255);
  for (int i = 0; i < count; i++) {
    h_in[i] = (unsigned)int_dist(engine)
            | (unsigned)int_dist(engine) <<  8
            | (unsigned)int_dist(engine) << 16
            | (unsigned)int_dist(engine) << 24;
  }
  Relu_reference(count, h_in.data(), r_out.data());

  int *d_in  = new int[count];
  int *d_out = new int[count];
  memcpy(d_in, h_in.data(), count * sizeof(int));

#pragma omp target enter data map(to: d_in[0:count]) map(alloc: d_out[0:count])

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
#pragma omp target teams distribute parallel for thread_limit(256)
    for (int idx = 0; idx < count; idx++) {
      int v = d_in[idx];
      auto b0 = (signed char)( v        & 0xFF);
      auto b1 = (signed char)((v >>  8) & 0xFF);
      auto b2 = (signed char)((v >> 16) & 0xFF);
      auto b3 = (signed char)((v >> 24) & 0xFF);
      unsigned x = (unsigned)(b0 > 0 ? b0 : 0);
      unsigned y = (unsigned)(b1 > 0 ? b1 : 0);
      unsigned z = (unsigned)(b2 > 0 ? b2 : 0);
      unsigned w = (unsigned)(b3 > 0 ? b3 : 0);
      d_out[idx] = (int)(w << 24 | z << 16 | y << 8 | x);
    }
  }
  end  = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of Relu_impl1 Kernel : %f (us)\n", (time * 1e-3f) / repeat);

#pragma omp target update from(d_out[0:count])
  for (int i = 0; i < count; i++) h_out[i] = d_out[i];
  printf("%s\n", memcmp(h_out.data(), r_out.data(), count * sizeof(int)) ? "FAIL" : "PASS");

  // impl2
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
#pragma omp target teams distribute parallel for thread_limit(256)
    for (int idx = 0; idx < count; idx++) {
      int v = d_in[idx];
      auto b0 = (signed char)( v        & 0xFF);
      auto b1 = (signed char)((v >>  8) & 0xFF);
      auto b2 = (signed char)((v >> 16) & 0xFF);
      auto b3 = (signed char)((v >> 24) & 0xFF);
      unsigned x = (unsigned)(b0 > 0 ? b0 : 0);
      unsigned y = (unsigned)(b1 > 0 ? b1 : 0);
      unsigned z = (unsigned)(b2 > 0 ? b2 : 0);
      unsigned w = (unsigned)(b3 > 0 ? b3 : 0);
      d_out[idx] = (int)(w << 24 | z << 16 | y << 8 | x);
    }
  }
  end  = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of Relu_impl2 Kernel: %f (us)\n", (time * 1e-3f) / repeat);

#pragma omp target update from(d_out[0:count])
  for (int i = 0; i < count; i++) h_out[i] = d_out[i];
  printf("%s\n", memcmp(h_out.data(), r_out.data(), count * sizeof(int)) ? "FAIL" : "PASS");

#pragma omp target exit data map(delete: d_in[0:count], d_out[0:count])
  delete[] d_in; delete[] d_out;
  return 0;
}
