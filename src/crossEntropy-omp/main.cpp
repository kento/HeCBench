/*  Copyright (c) 2021-2022 Intel Corporation
   Licensed under the Apache License, Version 2.0 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

#define random_int(a, b) ( rand() % ((b) - (a)) + (a) )
#define random_float() (rand() / static_cast<double>(RAND_MAX))

#define tolerance 4e-3f

constexpr int bs = 128;
constexpr int W  = 81;
constexpr int H  = 8732;
constexpr int PredictShape = bs * W * H;
constexpr int TargetShape  = bs * H;
constexpr int OutputShape  = bs * H;

void loss_bwd_cpu(
    const float*   log_softmax,
    const int64_t* target,
    const float*   weight,
    const int64_t* mask,
    const float*   grad_output,
    const float*   grad_output_neg,
          float*   grad_predict)
{
  for (int i = 0; i < bs; ++i) {
    for (int k = 0; k < W; ++k) {
      for (int j = 0; j < H; ++j) {
        int64_t offset2d      = (int64_t)i * H + j;
        int64_t idx           = target[offset2d];
        int64_t predict_offset = (int64_t)i * W * H + (int64_t)k * H + j;
        if (idx == (int64_t)k) {
          grad_predict[predict_offset] =
            (-grad_output[offset2d] * weight[offset2d]) +
            (mask[offset2d] ? -grad_output_neg[offset2d] * weight[offset2d] : 0.0f);
        } else {
          grad_predict[predict_offset] = 0.0f;
        }
      }
    }
  }

  std::vector<float> sum_value((size_t)bs * H, 0.0f);
  for (int i = 0; i < bs; ++i)
    for (int k = 0; k < W; ++k)
      for (int j = 0; j < H; ++j) {
        int64_t offset = (int64_t)i * W * H + (int64_t)k * H + j;
        sum_value[(size_t)i * H + j] += grad_predict[offset] * log_softmax[offset];
      }

  for (int i = 0; i < bs; ++i)
    for (int k = 0; k < W; ++k)
      for (int j = 0; j < H; ++j) {
        int64_t offset = (int64_t)i * W * H + (int64_t)k * H + j;
        grad_predict[offset] -= std::exp(log_softmax[offset]) * sum_value[(size_t)i * H + j];
      }
}

static int64_t errors = 0;

void verify(const float* cpu, const float* dev, size_t sz) {
  int count = 0;
  for (size_t i = 0; i < sz; ++i) {
    if (std::abs(cpu[i] - dev[i]) > tolerance) {
      ++count;
      if (count < 10)
        std::cout << "Error at i=" << i << " cpu=" << cpu[i]
                  << " dev=" << dev[i] << " gap=" << (cpu[i] - dev[i]) << "\n";
    }
  }
  errors += count;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  printf("Tensor size (BatchSize * Width * Height) = %d * %d * %d\n", bs, W, H);
  printf("=========== Data type is FP32 ===========\n");

  srand(42);

  std::vector<float>   log_softmax(PredictShape);
  std::vector<float>   grad_output(OutputShape);
  std::vector<float>   grad_output_neg(OutputShape);
  std::vector<float>   weight(OutputShape);
  std::vector<int64_t> target(TargetShape);
  std::vector<int64_t> mask(TargetShape);
  std::vector<float>   grad_predict_cpu(PredictShape);
  std::vector<float>   grad_predict_dev(PredictShape, 0.0f);

  for (int i = 0; i < PredictShape; ++i) log_softmax[i]      = (float)random_float();
  for (int i = 0; i < OutputShape;  ++i) grad_output[i]      = (float)random_float();
  for (int i = 0; i < OutputShape;  ++i) grad_output_neg[i]  = (float)random_float();
  for (int i = 0; i < OutputShape;  ++i) weight[i]           = (float)random_float();
  for (int i = 0; i < TargetShape;  ++i) target[i]           = (int64_t)random_int(0, W - 1);
  for (int i = 0; i < TargetShape;  ++i) mask[i]             = (int64_t)random_int(0, 1);

  float*   d_log_softmax     = log_softmax.data();
  float*   d_grad_output     = grad_output.data();
  float*   d_grad_output_neg = grad_output_neg.data();
  int64_t* d_target          = target.data();
  float*   d_weight          = weight.data();
  int64_t* d_mask            = mask.data();
  float*   d_grad_predict    = grad_predict_dev.data();

  #pragma omp target enter data map(alloc: \
      d_log_softmax[0:PredictShape], d_grad_output[0:OutputShape], \
      d_grad_output_neg[0:OutputShape], d_target[0:TargetShape], \
      d_weight[0:OutputShape], d_mask[0:TargetShape], \
      d_grad_predict[0:PredictShape])
  #pragma omp target update to(\
      d_log_softmax[0:PredictShape], d_grad_output[0:OutputShape], \
      d_grad_output_neg[0:OutputShape], d_target[0:TargetShape], \
      d_weight[0:OutputShape], d_mask[0:TargetShape])

  // Warm-up
  for (int k = 0; k < 10; ++k) {
    #pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
    for (int b = 0; b < bs; b++) {
      for (int h = 0; h < H; h++) {
        int64_t offset2d = (int64_t)b * H + h;
        int64_t idx      = d_target[offset2d];
        float tmp_grad = -(d_grad_output[offset2d] +
                          (d_mask[offset2d] ? d_grad_output_neg[offset2d] : 0.0f)) *
                         d_weight[offset2d];
        float sum_value = tmp_grad *
          d_log_softmax[(int64_t)b * W * H + idx * H + h];
        for (int i = 0; i < W; ++i) {
          int64_t in_offset = (int64_t)b * W * H + (int64_t)i * H + h;
          float tmp_sfm = expf(d_log_softmax[in_offset]) * sum_value;
          d_grad_predict[in_offset] = (i == idx ? tmp_grad : 0.0f) - tmp_sfm;
        }
      }
    }
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < repeat; ++k) {
    #pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
    for (int b = 0; b < bs; b++) {
      for (int h = 0; h < H; h++) {
        int64_t offset2d = (int64_t)b * H + h;
        int64_t idx      = d_target[offset2d];
        float tmp_grad = -(d_grad_output[offset2d] +
                          (d_mask[offset2d] ? d_grad_output_neg[offset2d] : 0.0f)) *
                         d_weight[offset2d];
        float sum_value = tmp_grad *
          d_log_softmax[(int64_t)b * W * H + idx * H + h];
        for (int i = 0; i < W; ++i) {
          int64_t in_offset = (int64_t)b * W * H + (int64_t)i * H + h;
          float tmp_sfm = expf(d_log_softmax[in_offset]) * sum_value;
          d_grad_predict[in_offset] = (i == idx ? tmp_grad : 0.0f) - tmp_sfm;
        }
      }
    }
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double kernel_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  #pragma omp target update from(d_grad_predict[0:PredictShape])

  // CPU reference
  auto t_cpu0 = std::chrono::high_resolution_clock::now();
  loss_bwd_cpu(log_softmax.data(), target.data(), weight.data(),
               mask.data(), grad_output.data(), grad_output_neg.data(),
               grad_predict_cpu.data());
  auto t_cpu1 = std::chrono::high_resolution_clock::now();
  double cpu_time = std::chrono::duration<double, std::milli>(t_cpu1 - t_cpu0).count();

  verify(grad_predict_cpu.data(), grad_predict_dev.data(), (size_t)PredictShape);

  double avg_kernel_ms = kernel_time_ms / repeat;
  double allBytes =
    (double)sizeof(float)   * (double)(PredictShape * 2.0 + OutputShape * 3.0)
  + (double)sizeof(int64_t) * (double)(TargetShape  * 2.0);

  printf("Average kernel time (ms)   : %f\n", avg_kernel_ms);
  printf("CPU serial time (ms)       : %f\n", cpu_time);
  printf("BandWidth = %lf (GB/s)\n", allBytes / (avg_kernel_ms / 1000.0) / 1e9);
  printf("%s\n", (errors == 0) ? "PASS" : "FAIL");

  #pragma omp target exit data map(delete: \
      d_log_softmax[0:PredictShape], d_grad_output[0:OutputShape], \
      d_grad_output_neg[0:OutputShape], d_target[0:TargetShape], \
      d_weight[0:OutputShape], d_mask[0:TargetShape], \
      d_grad_predict[0:PredictShape])
  return 0;
}
