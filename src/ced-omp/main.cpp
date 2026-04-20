// Canny Edge Detection benchmark – OpenMP target offloading port
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <chrono>

static const float c_gaus[9] = {
  0.0625f, 0.125f, 0.0625f,
  0.1250f, 0.250f, 0.1250f,
  0.0625f, 0.125f, 0.0625f
};
static const int c_sobx[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
static const int c_soby[9] = { -1, -2, -1,  0, 0, 0,  1, 2, 1 };

void run_gaussian(
    const uint8_t* in, uint8_t* out,
    const float* gaus,
    int rows, int cols)
{
  #pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
  for (int row = 1; row < rows - 1; row++) {
    for (int col = 1; col < cols - 1; col++) {
      int sum = 0;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          sum += (int)(gaus[i * 3 + j] * (float)in[(row - 1 + i) * cols + (col - 1 + j)]);
      out[row * cols + col] = (uint8_t)(sum < 0 ? 0 : (sum > 255 ? 255 : sum));
    }
  }
}

void run_sobel(
    const uint8_t* in, uint8_t* out, uint8_t* theta,
    const int* sobx, const int* soby,
    int rows, int cols)
{
  #pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
  for (int row = 1; row < rows - 1; row++) {
    for (int col = 1; col < cols - 1; col++) {
      const float PI = 3.14159265f;
      float sumx = 0.f, sumy = 0.f;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
          float px = (float)in[(row - 1 + i) * cols + (col - 1 + j)];
          sumx += (float)sobx[i * 3 + j] * px;
          sumy += (float)soby[i * 3 + j] * px;
        }
      float mag = sqrtf(sumx * sumx + sumy * sumy);
      mag = mag < 0.f ? 0.f : (mag > 255.f ? 255.f : mag);
      out[row * cols + col] = (uint8_t)mag;

      float angle = atan2f(sumy, sumx);
      if (angle < 0.f)
        angle = fmodf(angle + 2.f * PI, 2.f * PI);

      uint8_t dir;
      if      (angle <= PI / 8.f)          dir = 0;
      else if (angle <= 3.f * PI / 8.f)    dir = 45;
      else if (angle <= 5.f * PI / 8.f)    dir = 90;
      else if (angle <= 7.f * PI / 8.f)    dir = 135;
      else if (angle <= 9.f * PI / 8.f)    dir = 0;
      else if (angle <= 11.f * PI / 8.f)   dir = 45;
      else if (angle <= 13.f * PI / 8.f)   dir = 90;
      else if (angle <= 15.f * PI / 8.f)   dir = 135;
      else                                  dir = 0;
      theta[row * cols + col] = dir;
    }
  }
}

void run_nms(
    const uint8_t* in, uint8_t* out,
    const uint8_t* theta,
    int rows, int cols)
{
  #pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
  for (int row = 1; row < rows - 1; row++) {
    for (int col = 1; col < cols - 1; col++) {
      const int pos = row * cols + col;
      uint8_t mag = in[pos];
      uint8_t dir = theta[pos];
      bool suppress;
      switch (dir) {
        case 0:
          suppress = (mag <= in[row * cols + col + 1]) ||
                     (mag <= in[row * cols + col - 1]);
          break;
        case 45:
          suppress = (mag <= in[(row - 1) * cols + col + 1]) ||
                     (mag <= in[(row + 1) * cols + col - 1]);
          break;
        case 90:
          suppress = (mag <= in[(row - 1) * cols + col]) ||
                     (mag <= in[(row + 1) * cols + col]);
          break;
        default:
          suppress = (mag <= in[(row - 1) * cols + col - 1]) ||
                     (mag <= in[(row + 1) * cols + col + 1]);
          break;
      }
      out[pos] = suppress ? 0 : mag;
    }
  }
}

void run_threshold(
    const uint8_t* in, uint8_t* out,
    int rows, int cols)
{
  const float lowThresh  = 10.f;
  const float highThresh = 70.f;
  const float med        = (highThresh + lowThresh) / 2.f;
  const uint8_t EDGE     = 255;

  #pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
  for (int row = 1; row < rows - 1; row++) {
    for (int col = 1; col < cols - 1; col++) {
      const int pos = row * cols + col;
      float mag = (float)in[pos];
      if      (mag >= highThresh)  out[pos] = EDGE;
      else if (mag <= lowThresh)   out[pos] = 0;
      else if (mag >= med)         out[pos] = EDGE;
      else                         out[pos] = 0;
    }
  }
}

int main(int argc, char* argv[]) {
  int rows   = 2048;
  int cols   = 2048;
  int repeat = 100;
  if (argc > 1) rows   = std::atoi(argv[1]);
  if (argc > 2) cols   = std::atoi(argv[2]);
  if (argc > 3) repeat = std::atoi(argv[3]);

  printf("Image: %d x %d, repeat: %d\n", rows, cols, repeat);

  const int N = rows * cols;

  uint8_t* h_input = (uint8_t*)malloc(N * sizeof(uint8_t));
  {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);
    for (int i = 0; i < N; ++i) h_input[i] = (uint8_t)dist(rng);
  }

  float* d_gaus  = (float*)  malloc(9 * sizeof(float));
  int*   d_sobx  = (int*)    malloc(9 * sizeof(int));
  int*   d_soby  = (int*)    malloc(9 * sizeof(int));
  for (int i = 0; i < 9; ++i) { d_gaus[i] = c_gaus[i]; d_sobx[i] = c_sobx[i]; d_soby[i] = c_soby[i]; }

  uint8_t* d_in    = (uint8_t*)malloc(N * sizeof(uint8_t));
  uint8_t* d_blur  = (uint8_t*)malloc(N * sizeof(uint8_t));
  uint8_t* d_grad  = (uint8_t*)malloc(N * sizeof(uint8_t));
  uint8_t* d_nms   = (uint8_t*)malloc(N * sizeof(uint8_t));
  uint8_t* d_out   = (uint8_t*)malloc(N * sizeof(uint8_t));
  uint8_t* d_theta = (uint8_t*)malloc(N * sizeof(uint8_t));
  memcpy(d_in, h_input, N * sizeof(uint8_t));

  #pragma omp target enter data map(alloc: d_gaus[0:9], d_sobx[0:9], d_soby[0:9], \
      d_in[0:N], d_blur[0:N], d_grad[0:N], d_nms[0:N], d_out[0:N], d_theta[0:N])
  #pragma omp target update to(d_gaus[0:9], d_sobx[0:9], d_soby[0:9], d_in[0:N])

  auto t_start = std::chrono::high_resolution_clock::now();

  for (int r = 0; r < repeat; ++r) {
    run_gaussian(d_in,   d_blur, d_gaus, rows, cols);
    run_sobel   (d_blur, d_grad, d_theta, d_sobx, d_soby, rows, cols);
    run_nms     (d_grad, d_nms,  d_theta, rows, cols);
    run_threshold(d_nms, d_out, rows, cols);
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1e6;
  printf("Total time: %.4f s  (%.4f ms/iter)\n", elapsed, elapsed / repeat * 1e3);

  #pragma omp target update from(d_out[0:N])
  int nonzero = 0;
  for (int i = 0; i < N; ++i) if (d_out[i] != 0) ++nonzero;
  if (nonzero == 0)
    printf("VERIFICATION FAILED: output is all zeros\n");
  else
    printf("PASS: %d non-zero edge pixels\n", nonzero);

  #pragma omp target exit data map(delete: d_gaus[0:9], d_sobx[0:9], d_soby[0:9], \
      d_in[0:N], d_blur[0:N], d_grad[0:N], d_nms[0:N], d_out[0:N], d_theta[0:N])
  free(d_gaus); free(d_sobx); free(d_soby);
  free(d_in); free(d_blur); free(d_grad); free(d_nms); free(d_out); free(d_theta);
  free(h_input);
  return 0;
}
