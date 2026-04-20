#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int beamsize    = 4;
  const int nhead       = 16;
  const int dim_feature = nhead * 256;
  const int n_steps     = 9;
  const float scaler    = sqrtf(nhead * 1.f / dim_feature);
  const int qk_col      = dim_feature;
  const int v_col       = dim_feature;
  const int THRESHOLD   = 64;
  const int nteams      = beamsize * nhead;
  const int dim_per_head = qk_col / nhead;

  int q_size = beamsize * dim_feature;
  int k_size = beamsize * dim_feature * n_steps;
  int v_size = beamsize * dim_feature * n_steps;

  float* d_q   = (float*)malloc(q_size * sizeof(float));
  float* d_k   = (float*)malloc(k_size * sizeof(float));
  float* d_v   = (float*)malloc(v_size * sizeof(float));
  float* d_dst = (float*)malloc(q_size * sizeof(float));
  // Per-team scratch for logits
  float* d_logits = (float*)malloc(nteams * n_steps * sizeof(float));

  #pragma omp target enter data map(alloc: d_q[0:q_size], d_k[0:k_size], d_v[0:v_size], d_dst[0:q_size], d_logits[0:nteams*n_steps])

  // Initialize on device
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < q_size; i++) d_q[i] = (float)(i % 100) / 100.f;
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < k_size; i++) d_k[i] = (float)(i % 100) / 100.f;
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < v_size; i++) d_v[i] = (float)(i % 100) / 100.f;

  auto start = std::chrono::steady_clock::now();

  for (int rep = 0; rep < repeat; rep++) {
    #pragma omp target teams distribute num_teams(nteams) thread_limit(256)
    for (int bid = 0; bid < nteams; bid++) {
      const int candidate_id = bid / nhead;
      const int head_id      = bid % nhead;

      // QK^T dot products → logits
      #pragma omp parallel for
      for (int step = 0; step < n_steps; step++) {
        float s = 0.f;
        const float* q_ptr = d_q + candidate_id * qk_col + head_id * dim_per_head;
        const float* k_ptr = d_k + candidate_id * qk_col * n_steps
                           + head_id * dim_per_head + step * qk_col;
        for (int i = 0; i < dim_per_head; i++) s += q_ptr[i] * k_ptr[i];
        d_logits[bid * n_steps + step] = s * scaler;
      }

      // Softmax: max reduction
      float max_val = -1e20f;
      #pragma omp parallel for reduction(max:max_val)
      for (int s = 0; s < n_steps; s++) {
        if (d_logits[bid * n_steps + s] > max_val)
          max_val = d_logits[bid * n_steps + s];
      }

      // Softmax: exp and sum
      float sum_exp = 0.f;
      #pragma omp parallel for reduction(+:sum_exp)
      for (int s = 0; s < n_steps; s++) {
        float diff = d_logits[bid * n_steps + s] - max_val;
        if (diff < -(float)THRESHOLD) diff = -(float)THRESHOLD;
        float e = expf(diff);
        d_logits[bid * n_steps + s] = e;
        sum_exp += e;
      }

      // Softmax: normalize
      #pragma omp parallel for
      for (int s = 0; s < n_steps; s++)
        d_logits[bid * n_steps + s] /= sum_exp;

      // Weighted sum of values → dst
      #pragma omp parallel for
      for (int feat = 0; feat < dim_per_head; feat++) {
        float s = 0.f;
        int base = candidate_id * v_col * n_steps + head_id * dim_per_head + feat;
        for (int step = 0; step < n_steps; step++)
          s += d_logits[bid * n_steps + step] * d_v[base + step * v_col];
        d_dst[candidate_id * v_col + head_id * dim_per_head + feat] = s;
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  #pragma omp target exit data map(from: d_dst[0:q_size]) \
    map(delete: d_q[0:q_size], d_k[0:k_size], d_v[0:v_size], d_logits[0:nteams*n_steps])

  for (int i = 0; i < beamsize - 1; i++) {
    float sum = 0.f;
    for (int j = 0; j < dim_feature; j++) {
      float d = d_dst[i * dim_feature + j] - d_dst[(i + 1) * dim_feature + j];
      sum += d * d;
    }
    printf("Distance between beams %d and %d: %f\n", i, i+1, sqrtf(sum));
  }

  free(d_q); free(d_k); free(d_v); free(d_dst); free(d_logits);
  return 0;
}
