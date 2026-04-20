/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 * OpenMP target offloading port.
 */

#include <chrono>
#include <cstdio>
#include <omp.h>

void addBiasQKV(int batch_size, int seq_len, int hidden_units, int head_num, int repeat) {
  int m = batch_size * seq_len;
  int size_per_head = hidden_units / head_num;
  int total_size = m * 3 * hidden_units;

  float *qkv_src = (float*)malloc(total_size * sizeof(float));
  float *qkv_tgt = (float*)malloc(total_size * sizeof(float));
  float *bias     = (float*)malloc(3 * hidden_units * sizeof(float));
  float input_scale  = 1.0f;
  float output_scale = 1.0f;

  for (int i = 0; i < total_size; i++)     qkv_src[i] = 0.5f;
  for (int i = 0; i < 3*hidden_units; i++) bias[i]    = 0.1f;

  #pragma omp target enter data map(to: qkv_src[0:total_size], bias[0:3*hidden_units]) \
                                map(alloc: qkv_tgt[0:total_size])

  int N = m * 3 * head_num * size_per_head;

  // Warmup
  for (int r = 0; r < repeat; r++) {
    #pragma omp target teams distribute parallel for thread_limit(256) \
      map(to: input_scale, output_scale)
    for (int idx = 0; idx < N; idx++) {
      int feat    = idx % size_per_head;
      int tmp     = idx / size_per_head;
      int head_id = tmp % head_num;
      int tmp2    = tmp / head_num;
      int qkv_idx = tmp2 % 3;
      int word_id = tmp2 / 3;

      int src_id  = word_id * 3 * hidden_units + qkv_idx * hidden_units
                  + head_id * size_per_head + feat;
      int bias_id = qkv_idx * hidden_units + head_id * size_per_head + feat;
      int tgt_id  = word_id * 3 * hidden_units + head_id * 3 * size_per_head
                  + qkv_idx * size_per_head + feat;

      float val = (qkv_src[src_id] * input_scale + bias[bias_id]) * output_scale;
      qkv_tgt[tgt_id] = val;
    }
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < repeat; r++) {
    #pragma omp target teams distribute parallel for thread_limit(256) \
      map(to: input_scale, output_scale)
    for (int idx = 0; idx < N; idx++) {
      int feat    = idx % size_per_head;
      int tmp     = idx / size_per_head;
      int head_id = tmp % head_num;
      int tmp2    = tmp / head_num;
      int qkv_idx = tmp2 % 3;
      int word_id = tmp2 / 3;

      int src_id  = word_id * 3 * hidden_units + qkv_idx * hidden_units
                  + head_id * size_per_head + feat;
      int bias_id = qkv_idx * hidden_units + head_id * size_per_head + feat;
      int tgt_id  = word_id * 3 * hidden_units + head_id * 3 * size_per_head
                  + qkv_idx * size_per_head + feat;

      float val = (qkv_src[src_id] * input_scale + bias[bias_id]) * output_scale;
      qkv_tgt[tgt_id] = val;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the kernel: %f (us)\n", (time * 1e-3f) / repeat);

  #pragma omp target exit data map(delete: qkv_src[0:total_size], bias[0:3*hidden_units], \
                                           qkv_tgt[0:total_size])
  free(qkv_src);
  free(qkv_tgt);
  free(bias);
}

int main(int argc, char **argv) {
  int batch_size = 8;
  int seq_len = 1024;
  int hidden_units = 768;
  int head_num = 12;
  int repeat = 1000;
  addBiasQKV(batch_size, seq_len, hidden_units, head_num, repeat);
  return 0;
}
