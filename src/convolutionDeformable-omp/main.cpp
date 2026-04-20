// Deformable Convolution benchmark – OpenMP target offloading port
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

#pragma omp declare target
float bilinear_interp(const float* data, float h, float w, int height, int width) {
  if (h <= -1.f || h >= height || w <= -1.f || w >= width) return 0.f;
  int h0 = (int)floorf(h), h1 = h0 + 1;
  int w0 = (int)floorf(w), w1 = w0 + 1;
  float lh = h - h0, lw = w - w0;
  float hh = 1.f - lh,  hw = 1.f - lw;
  float v1 = (h0 >= 0 && w0 >= 0)        ? data[h0 * width + w0] : 0.f;
  float v2 = (h0 >= 0 && w1 < width)     ? data[h0 * width + w1] : 0.f;
  float v3 = (h1 < height && w0 >= 0)    ? data[h1 * width + w0] : 0.f;
  float v4 = (h1 < height && w1 < width) ? data[h1 * width + w1] : 0.f;
  return hh * hw * v1 + hh * lw * v2 + lh * hw * v3 + lh * lw * v4;
}
#pragma omp end declare target

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  constexpr int batch            = 1;
  constexpr int channels_in      = 64;
  constexpr int channels_out     = 64;
  constexpr int height           = 28;
  constexpr int width            = 28;
  constexpr int kernel_h         = 3;
  constexpr int kernel_w         = 3;
  constexpr int stride_h         = 1;
  constexpr int stride_w         = 1;
  constexpr int pad_h            = 1;
  constexpr int pad_w            = 1;
  constexpr int dilation_h       = 1;
  constexpr int dilation_w       = 1;
  constexpr int deformable_groups = 1;

  constexpr int height_out = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
  constexpr int width_out  = (width  + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

  constexpr int ksize      = kernel_h * kernel_w;
  constexpr int col_rows   = channels_in * ksize;
  constexpr int col_cols   = height_out * width_out;
  constexpr int n_offset   = 2 * deformable_groups * ksize;
  constexpr int n_mask     = deformable_groups * ksize;

  const int input_size  = batch * channels_in * height * width;
  const int offset_size = batch * n_offset * height_out * width_out;
  const int mask_size   = batch * n_mask   * height_out * width_out;
  const int weight_size = channels_out * (channels_in / deformable_groups) * ksize;
  const int output_size = batch * channels_out * height_out * width_out;
  const int columns_size= col_rows * col_cols;

  float* input   = (float*)malloc(input_size   * sizeof(float));
  float* offset  = (float*)malloc(offset_size  * sizeof(float));
  float* mask    = (float*)malloc(mask_size    * sizeof(float));
  float* weight  = (float*)malloc(weight_size  * sizeof(float));
  float* bias    = (float*)malloc(channels_out * sizeof(float));
  float* columns = (float*)malloc(columns_size * sizeof(float));
  float* output  = (float*)malloc(output_size  * sizeof(float));

  srand(42);
  for (int i = 0; i < input_size;   i++) input [i] = ((float)rand() / RAND_MAX - 0.5f);
  for (int i = 0; i < offset_size;  i++) offset[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
  for (int i = 0; i < mask_size;    i++) mask  [i] = 0.9f + 0.2f * (float)rand() / RAND_MAX;
  for (int i = 0; i < weight_size;  i++) weight[i] = ((float)rand() / RAND_MAX - 0.5f);
  for (int i = 0; i < channels_out; i++) bias  [i] = ((float)rand() / RAND_MAX - 0.5f);

  #pragma omp target enter data map(alloc: input[0:input_size], offset[0:offset_size], \
      mask[0:mask_size], weight[0:weight_size], bias[0:channels_out], \
      columns[0:columns_size], output[0:output_size])
  #pragma omp target update to(input[0:input_size], offset[0:offset_size], \
      mask[0:mask_size], weight[0:weight_size], bias[0:channels_out])

  // Warm-up im2col
  const int total_im2col = batch * channels_in * ksize * height_out * width_out;
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int idx = 0; idx < total_im2col; idx++) {
    int tmp   = idx;
    int w_out = tmp % width_out;  tmp /= width_out;
    int h_out = tmp % height_out; tmp /= height_out;
    int kw    = tmp % kernel_w;   tmp /= kernel_w;
    int kh    = tmp % kernel_h;   tmp /= kernel_h;
    int c_in  = tmp % channels_in; tmp /= channels_in;
    int b     = tmp;

    int offset_group = c_in / (channels_in / deformable_groups);
    int kp           = kh * kernel_w + kw;
    int offset_idx   = offset_group * 2 * ksize + kp * 2;
    int mask_idx     = offset_group * ksize + kp;
    int spatial      = h_out * width_out + w_out;
    int spatial_sz   = height_out * width_out;

    float h_in = h_out * stride_h - pad_h + kh * dilation_h
      + offset[b * n_offset * spatial_sz + offset_idx       * spatial_sz + spatial];
    float w_in = w_out * stride_w - pad_w + kw * dilation_w
      + offset[b * n_offset * spatial_sz + (offset_idx + 1) * spatial_sz + spatial];
    float mask_val = mask[(b * n_mask + mask_idx) * spatial_sz + spatial];

    const float* input_ptr = input + (b * channels_in + c_in) * height * width;
    float val = bilinear_interp(input_ptr, h_in, w_in, height, width) * mask_val;
    columns[(c_in * ksize + kp) * col_cols + spatial] = val;
  }

  auto t0 = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int idx = 0; idx < total_im2col; idx++) {
      int tmp   = idx;
      int w_out = tmp % width_out;  tmp /= width_out;
      int h_out = tmp % height_out; tmp /= height_out;
      int kw    = tmp % kernel_w;   tmp /= kernel_w;
      int kh    = tmp % kernel_h;   tmp /= kernel_h;
      int c_in  = tmp % channels_in; tmp /= channels_in;
      int b     = tmp;

      int offset_group = c_in / (channels_in / deformable_groups);
      int kp           = kh * kernel_w + kw;
      int offset_idx   = offset_group * 2 * ksize + kp * 2;
      int mask_idx     = offset_group * ksize + kp;
      int spatial      = h_out * width_out + w_out;
      int spatial_sz   = height_out * width_out;

      float h_in = h_out * stride_h - pad_h + kh * dilation_h
        + offset[b * n_offset * spatial_sz + offset_idx       * spatial_sz + spatial];
      float w_in = w_out * stride_w - pad_w + kw * dilation_w
        + offset[b * n_offset * spatial_sz + (offset_idx + 1) * spatial_sz + spatial];
      float mask_val = mask[(b * n_mask + mask_idx) * spatial_sz + spatial];

      const float* input_ptr = input + (b * channels_in + c_in) * height * width;
      float val = bilinear_interp(input_ptr, h_in, w_in, height, width) * mask_val;
      columns[(c_in * ksize + kp) * col_cols + spatial] = val;
    }
  }

  auto t1 = std::chrono::steady_clock::now();
  double im2col_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / repeat;
  printf("im2col average time: %.4f ms\n", im2col_ms);

  // matmul
  #pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
  for (int c_out = 0; c_out < channels_out; c_out++) {
    for (int col = 0; col < col_cols; col++) {
      float sum = bias[c_out];
      for (int k = 0; k < col_rows; k++)
        sum += weight[c_out * col_rows + k] * columns[k * col_cols + col];
      output[c_out * col_cols + col] = sum;
    }
  }

  #pragma omp target update from(output[0:output_size])
  double sum = 0.0;
  for (int i = 0; i < output_size; i++) sum += output[i];
  printf("Output sum: %f\n", sum);
  printf("%s\n", (sum != 0.0) ? "PASS" : "FAIL");

  #pragma omp target exit data map(delete: input[0:input_size], offset[0:offset_size], \
      mask[0:mask_size], weight[0:weight_size], bias[0:channels_out], \
      columns[0:columns_size], output[0:output_size])
  free(input); free(offset); free(mask); free(weight); free(bias); free(columns); free(output);
  return 0;
}
