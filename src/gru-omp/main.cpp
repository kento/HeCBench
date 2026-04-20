// OpenMP target offloading port of gru-kokkos (GRU cell forward pass)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <omp.h>

#pragma omp declare target
static inline float gru_sigmoid(float x)
{
  return 1.f / (1.f + expf(-x));
}
#pragma omp end declare target

int main(int argc, char *argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of sequences> <hidden size> <repeat>\n", argv[0]);
    return 1;
  }
  const int vsz    = atoi(argv[1]);
  const int hsz    = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const int input_size  = 3 * vsz * hsz;
  const int hidden_size = 3 * vsz * hsz;
  const int bias_size   = 3 * hsz;
  const int store_size  = 5 * vsz * hsz;
  const int state_size  = vsz;

  float *h_input        = new float[input_size];
  float *h_hidden       = new float[hidden_size];
  float *h_input_bias   = new float[bias_size];
  float *h_hidden_bias  = new float[bias_size];
  float *h_hx           = new float[state_size];

  std::default_random_engine g(123);
  std::uniform_real_distribution<float> distr(-2.f, 2.f);

  for (int i = 0; i < input_size;  i++) h_input[i]       = distr(g);
  for (int i = 0; i < hidden_size; i++) h_hidden[i]      = distr(g);
  for (int i = 0; i < bias_size;   i++) { h_input_bias[i] = distr(g);
                                           h_hidden_bias[i] = distr(g); }
  for (int i = 0; i < state_size;  i++) h_hx[i] = distr(g);

  float* d_input       = (float*)malloc(input_size  * sizeof(float));
  float* d_hidden      = (float*)malloc(hidden_size * sizeof(float));
  float* d_input_bias  = (float*)malloc(bias_size   * sizeof(float));
  float* d_hidden_bias = (float*)malloc(bias_size   * sizeof(float));
  float* d_hx          = (float*)malloc(state_size  * sizeof(float));
  float* d_hy          = (float*)malloc(state_size  * sizeof(float));
  float* d_store       = (float*)malloc(store_size  * sizeof(float));

  for (int i = 0; i < input_size;  i++) d_input[i]       = h_input[i];
  for (int i = 0; i < hidden_size; i++) d_hidden[i]      = h_hidden[i];
  for (int i = 0; i < bias_size;   i++) { d_input_bias[i] = h_input_bias[i];
                                           d_hidden_bias[i] = h_hidden_bias[i]; }
  for (int i = 0; i < state_size;  i++) d_hx[i] = h_hx[i];

  #pragma omp target enter data \
    map(to: d_input[0:input_size], d_hidden[0:hidden_size], \
            d_input_bias[0:bias_size], d_hidden_bias[0:bias_size], d_hx[0:state_size]) \
    map(alloc: d_hy[0:state_size], d_store[0:store_size])

  auto start = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    const int hsz_ = hsz;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int linearIndex = 0; linearIndex < vsz; linearIndex++) {
      const int offset = (linearIndex / hsz_) * 3 * hsz_ + linearIndex % hsz_;

      const float ir = d_input[offset + 0 * hsz_];
      const float ii = d_input[offset + 1 * hsz_];
      const float in = d_input[offset + 2 * hsz_];
      const float hr = d_hidden[offset + 0 * hsz_];
      const float hi = d_hidden[offset + 1 * hsz_];
      const float hn = d_hidden[offset + 2 * hsz_];
      const float hx = d_hx[linearIndex];

      const int bi = linearIndex % hsz_;
      const float b1r = d_input_bias[bi + 0 * hsz_];
      const float b1i = d_input_bias[bi + 1 * hsz_];
      const float b1n = d_input_bias[bi + 2 * hsz_];
      const float b2r = d_hidden_bias[bi + 0 * hsz_];
      const float b2i = d_hidden_bias[bi + 1 * hsz_];
      const float b2n = d_hidden_bias[bi + 2 * hsz_];

      const float rg = gru_sigmoid(ir + hr + b1r + b2r);
      const float ig = gru_sigmoid(ii + hi + b1i + b2i);
      const float ng = tanhf(in + b1n + rg * (hn + b2n));

      d_hy[linearIndex] = ng + ig * (hx - ng);

      const int soffset = (linearIndex / hsz_) * 5 * hsz_ + linearIndex % hsz_;
      d_store[soffset + 0 * hsz_] = rg;
      d_store[soffset + 1 * hsz_] = ig;
      d_store[soffset + 2 * hsz_] = ng;
      d_store[soffset + 3 * hsz_] = hx;
      d_store[soffset + 4 * hsz_] = hn + b2n;
    }
  }

  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of gru_cell_forward: %f (us)\n",
         (time * 1e-3f) / repeat);

  #pragma omp target update from(d_hy[0:state_size])

  float checksum = 0.f;
  for (int i = 0; i < state_size; i++) checksum += d_hy[i];
  printf("Checksum is %f\n", checksum / state_size);

  #pragma omp target exit data \
    map(delete: d_input[0:input_size], d_hidden[0:hidden_size], \
                d_input_bias[0:bias_size], d_hidden_bias[0:bias_size], \
                d_hx[0:state_size], d_hy[0:state_size], d_store[0:store_size])

  free(d_input); free(d_hidden); free(d_input_bias); free(d_hidden_bias);
  free(d_hx); free(d_hy); free(d_store);
  delete[] h_input; delete[] h_hidden; delete[] h_input_bias;
  delete[] h_hidden_bias; delete[] h_hx;
  return 0;
}
