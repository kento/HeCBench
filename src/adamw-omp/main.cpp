#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <omp.h>

#pragma omp declare target
float q_mapping(const float* __restrict__ qmap,
                const float* __restrict__ qmidpt,
                float x)
{
  int low = 0;
  int high = 15;

  if (x <= qmap[low]) return low;
  if (qmap[high] <= x) return high;

  while (low < high) {
    int mid = (low + high) >> 1;
    if (qmap[mid] <= x)
      low = mid + 1;
    else
      high = mid;
  }
  return (qmidpt[low-1] < x) ? low : low-1;
}
#pragma omp end declare target

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <vector size> <number of time steps>\n", argv[0]);
    return 1;
  }

  const long vector_size = atol(argv[1]);
  const int time_step    = atoi(argv[2]);

  int64_t size_bytes = vector_size * 2 * sizeof(float);

  float*  g        = (float*)  malloc(size_bytes);
  float*  p        = (float*)  malloc(size_bytes);
  float*  m_qscale = (float*)  malloc(size_bytes);
  float*  v_qscale = (float*)  malloc(size_bytes);
  int8_t* m        = (int8_t*) malloc(vector_size);
  int8_t* v        = (int8_t*) malloc(vector_size);
  float*  r        = (float*)  malloc(size_bytes);

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dist(0, 1);
  for (int64_t i = 0; i < vector_size * 2; i++) {
    m_qscale[i] = dist(gen);
    v_qscale[i] = dist(gen);
    g[i] = dist(gen);
    r[i] = p[i] = dist(gen);
  }
  for (int64_t i = 0; i < vector_size; i++) {
    m[i] = (int8_t)(256 * dist(gen));
    v[i] = (int8_t)(256 * dist(gen));
  }

  const float h_exp_qmap[16] = {
    -0.8875f, -0.6625f, -0.4375f, -0.2125f,
    -0.0775f, -0.0325f, -0.0055f,  0.0000f,
     0.0055f,  0.0325f,  0.0775f,  0.2125f,
     0.4375f,  0.6625f,  0.8875f,  1.0000f
  };
  const float h_exp_qmidpt[15] = {
    -0.775f, -0.55f, -0.325f, -0.145f, -0.055f,
    -0.019f, -0.00275f, 0.00275f, 0.019f, 0.055f,
     0.145f,  0.325f,  0.55f,  0.775f,  0.94375f
  };
  const float h_sq_qmap[16] = {
    0.0625f, 0.1250f, 0.1875f, 0.2500f,
    0.3125f, 0.3750f, 0.4375f, 0.5000f,
    0.5625f, 0.6250f, 0.6875f, 0.7500f,
    0.8125f, 0.8750f, 0.9375f, 1.0000f
  };
  const float h_sq_qmidpt[15] = {
    0.09375f, 0.15625f, 0.21875f, 0.28125f,
    0.34375f, 0.40625f, 0.46875f, 0.53125f,
    0.59375f, 0.65625f, 0.71875f, 0.78125f,
    0.84375f, 0.90625f, 0.96875f
  };

  const int threadsPerBlock = 64;
  const long nblocks = (vector_size + threadsPerBlock - 1) / threadsPerBlock;

  const float lr = 1e-3f;
  const float weight_decay = 1e-2f;
  const float beta1 = 0.9f;
  const float beta2 = 0.999f;
  const float eps = 1e-8f;
  const float resid_beta1 = 1.0f - beta1;
  const float resid_beta2 = 1.0f - beta2;
  const float weight_decay_update = 1.0f - lr * weight_decay;

  // Allocate device memory
  float*  d_p           = (float*)  malloc(vector_size * 2 * sizeof(float));
  float*  d_g           = (float*)  malloc(vector_size * 2 * sizeof(float));
  float*  d_exp_qscale  = (float*)  malloc(vector_size * 2 * sizeof(float));
  float*  d_sq_qscale   = (float*)  malloc(vector_size * 2 * sizeof(float));
  int8_t* d_exp         = (int8_t*) malloc(vector_size * sizeof(int8_t));
  int8_t* d_sq          = (int8_t*) malloc(vector_size * sizeof(int8_t));
  float*  d_exp_qmap    = (float*)  malloc(16 * sizeof(float));
  float*  d_exp_qmidpt  = (float*)  malloc(15 * sizeof(float));
  float*  d_sq_qmap     = (float*)  malloc(16 * sizeof(float));
  float*  d_sq_qmidpt   = (float*)  malloc(15 * sizeof(float));

  // Intermediate arrays for inter-thread communication
  float*  tmp_exp_left  = (float*)  malloc(nblocks * threadsPerBlock * sizeof(float));
  float*  tmp_exp_right = (float*)  malloc(nblocks * threadsPerBlock * sizeof(float));
  float*  tmp_sq_left   = (float*)  malloc(nblocks * threadsPerBlock * sizeof(float));
  float*  tmp_sq_right  = (float*)  malloc(nblocks * threadsPerBlock * sizeof(float));

  memcpy(d_p,          p,          vector_size * 2 * sizeof(float));
  memcpy(d_g,          g,          vector_size * 2 * sizeof(float));
  memcpy(d_exp_qscale, m_qscale,   vector_size * 2 * sizeof(float));
  memcpy(d_sq_qscale,  v_qscale,   vector_size * 2 * sizeof(float));
  memcpy(d_exp,        m,          vector_size * sizeof(int8_t));
  memcpy(d_sq,         v,          vector_size * sizeof(int8_t));
  memcpy(d_exp_qmap,   h_exp_qmap,   16 * sizeof(float));
  memcpy(d_exp_qmidpt, h_exp_qmidpt, 15 * sizeof(float));
  memcpy(d_sq_qmap,    h_sq_qmap,    16 * sizeof(float));
  memcpy(d_sq_qmidpt,  h_sq_qmidpt,  15 * sizeof(float));

  long n2    = vector_size * 2;
  long ntmp  = nblocks * threadsPerBlock;

  #pragma omp target enter data \
    map(to: d_p[0:n2], d_g[0:n2], d_exp_qscale[0:n2], d_sq_qscale[0:n2]) \
    map(to: d_exp[0:vector_size], d_sq[0:vector_size]) \
    map(to: d_exp_qmap[0:16], d_exp_qmidpt[0:15], d_sq_qmap[0:16], d_sq_qmidpt[0:15]) \
    map(alloc: tmp_exp_left[0:ntmp], tmp_exp_right[0:ntmp], \
               tmp_sq_left[0:ntmp],  tmp_sq_right[0:ntmp])

  auto start = std::chrono::steady_clock::now();

  for (int step = 1; step <= time_step; step++) {
    const float correction1       = 1.0f - powf(beta1, step);
    const float correction2_sqrt  = sqrtf(1.0f - powf(beta2, step));
    const float step_size         = lr / correction1;

    // Kernel 1: compute per-element updates and intermediate values
    #pragma omp target teams distribute parallel for thread_limit(64)
    for (long global_id = 0; global_id < nblocks * threadsPerBlock; global_id++) {
      const bool active = global_id < vector_size;
      const long block_id = global_id / threadsPerBlock;

      const uint8_t bitmask            = 15;
      const uint8_t right_pack_bitmask = (uint8_t)(bitmask << 4);

      float exp_left = 0.f, exp_right = 0.f;
      float sq_left  = 0.f, sq_right  = 0.f;

      if (active) {
        const int8_t exp_full = d_exp[global_id];
        const int8_t sq_full  = d_sq[global_id];

        float p_x = d_p[global_id * 2];
        float p_y = d_p[global_id * 2 + 1];
        const float g_x = d_g[global_id * 2];
        const float g_y = d_g[global_id * 2 + 1];

        const int8_t exp_left_index = exp_full & bitmask;
        const int8_t sq_left_index  = sq_full  & bitmask;

        p_x = p_x * weight_decay_update;

        float exp_avg_qscale = d_exp_qscale[block_id];

        exp_left = d_exp_qmap[exp_left_index] * exp_avg_qscale;
        exp_left = beta1 * exp_left + resid_beta1 * g_x;

        sq_left = d_sq_qmap[sq_left_index] * d_sq_qscale[block_id];
        sq_left = beta2 * sq_left + resid_beta2 * (g_x * g_x);

        d_p[global_id * 2] = p_x - (step_size * (exp_left / (sqrtf(sq_left) / correction2_sqrt + eps)));

        const int8_t exp_right_index = (exp_full >> 4) & bitmask;
        const int8_t sq_right_index  = (sq_full  >> 4) & bitmask;

        p_y = p_y * weight_decay_update;

        exp_right = d_exp_qmap[exp_right_index] * exp_avg_qscale;
        exp_right = beta1 * exp_right + resid_beta1 * g_y;

        sq_right = d_sq_qmap[sq_right_index] * d_sq_qscale[block_id];
        sq_right = beta2 * sq_right + resid_beta2 * (g_y * g_y);

        d_p[global_id * 2 + 1] = p_y - (step_size * (exp_right / (sqrtf(sq_right) / correction2_sqrt + eps)));
      }

      tmp_exp_left[global_id]  = exp_left;
      tmp_exp_right[global_id] = exp_right;
      tmp_sq_left[global_id]   = sq_left;
      tmp_sq_right[global_id]  = sq_right;
    }

    // Kernel 2: per-block max reduction
    #pragma omp target teams distribute num_teams(nblocks) thread_limit(64)
    for (long block_id = 0; block_id < nblocks; block_id++) {
      float absmax_exp = 0.f;
      float absmax_sq  = 0.f;
      #pragma omp parallel for reduction(max:absmax_exp,absmax_sq)
      for (int tid = 0; tid < threadsPerBlock; tid++) {
        long gid = block_id * threadsPerBlock + tid;
        if (gid < vector_size) {
          float ve = tmp_exp_left[gid] > tmp_exp_right[gid]
                   ? tmp_exp_left[gid] : tmp_exp_right[gid];
          float vs = tmp_sq_left[gid]  > tmp_sq_right[gid]
                   ? tmp_sq_left[gid]  : tmp_sq_right[gid];
          if (ve > absmax_exp) absmax_exp = ve;
          if (vs > absmax_sq)  absmax_sq  = vs;
        }
      }
      if (omp_get_thread_num() == 0) {
        d_exp_qscale[block_id] = absmax_exp;
        d_sq_qscale[block_id]  = absmax_sq;
      }
    }

    // Kernel 3: quantize and pack
    #pragma omp target teams distribute parallel for thread_limit(64)
    for (long global_id = 0; global_id < nblocks * threadsPerBlock; global_id++) {
      if (global_id >= vector_size) continue;

      const long block_id  = global_id / threadsPerBlock;
      const uint8_t bitmask = 15;

      float absmax_exp = d_exp_qscale[block_id];
      float absmax_sq  = d_sq_qscale[block_id];

      float exp_left  = tmp_exp_left[global_id];
      float exp_right = tmp_exp_right[global_id];
      float sq_left   = tmp_sq_left[global_id];
      float sq_right  = tmp_sq_right[global_id];

      int8_t local_packed_exp = 0;
      int8_t local_packed_sq  = 0;

      const int8_t q_exp_left  = (int8_t)q_mapping(d_exp_qmap, d_exp_qmidpt,
                                                     exp_left  / absmax_exp);
      const int8_t q_sq_left   = (int8_t)q_mapping(d_sq_qmap,  d_sq_qmidpt,
                                                     sq_left   / absmax_sq);
      local_packed_exp |= (q_exp_left & bitmask);
      local_packed_sq  |= (q_sq_left  & bitmask);

      const int8_t q_exp_right = (int8_t)q_mapping(d_exp_qmap, d_exp_qmidpt,
                                                     exp_right / absmax_exp);
      const int8_t q_sq_right  = (int8_t)q_mapping(d_sq_qmap,  d_sq_qmidpt,
                                                     sq_right  / absmax_sq);
      local_packed_exp |= (int8_t)((q_exp_right & bitmask) << 4);
      local_packed_sq  |= (int8_t)((q_sq_right  & bitmask) << 4);

      d_exp[global_id] = local_packed_exp;
      d_sq[global_id]  = local_packed_sq;
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (ms)\n", time * 1e-6f / time_step);

  #pragma omp target update from(d_p[0:n2])
  #pragma omp target exit data \
    map(delete: d_p[0:n2], d_g[0:n2], d_exp_qscale[0:n2], d_sq_qscale[0:n2]) \
    map(delete: d_exp[0:vector_size], d_sq[0:vector_size]) \
    map(delete: d_exp_qmap[0:16], d_exp_qmidpt[0:15], d_sq_qmap[0:16], d_sq_qmidpt[0:15]) \
    map(delete: tmp_exp_left[0:ntmp], tmp_exp_right[0:ntmp], \
                tmp_sq_left[0:ntmp],  tmp_sq_right[0:ntmp])

  memcpy(p, d_p, vector_size * 2 * sizeof(float));

  free(p); free(m_qscale); free(v_qscale); free(m); free(v); free(g); free(r);
  free(d_p); free(d_g); free(d_exp_qscale); free(d_sq_qscale);
  free(d_exp); free(d_sq);
  free(d_exp_qmap); free(d_exp_qmidpt); free(d_sq_qmap); free(d_sq_qmidpt);
  free(tmp_exp_left); free(tmp_exp_right); free(tmp_sq_left); free(tmp_sq_right);
  return 0;
}
