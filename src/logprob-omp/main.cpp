// OpenMP target port of logprob-kokkos: per-token log-probabilities.

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <limits>
#include <random>
#include <vector>

static void log_probs_cpu(
    float*       log_probs,
    const float* logits,
    const int*   ids,
    const int*   lengths,
    int          max_input_length,
    int          batch_size,
    int          vocab_size,
    int          vocab_size_padded,
    float*       cum_log_probs)
{
  for (int bidx = 0; bidx < batch_size; ++bidx) {
    float accum = 0.f;
    for (int step = 0; step < lengths[bidx] - 1; ++step) {
      int         batch_offset = bidx * max_input_length * vocab_size_padded;
      int         step_offset  = step * vocab_size_padded;
      const float* lptr        = logits + batch_offset + step_offset;

      float max_val = -std::numeric_limits<float>::max();
      for (int i = 0; i < vocab_size; ++i)
        max_val = std::fmaxf(max_val, lptr[i]);

      float sum_exp = 0.f;
      for (int i = 0; i < vocab_size; ++i)
        sum_exp += expf(lptr[i] - max_val);

      int idx        = step + bidx * (max_input_length - 1);
      int token_idx  = step + 1 + bidx * max_input_length;
      log_probs[idx] = lptr[ids[token_idx]] - max_val - logf(sum_exp + 1e-9f);
      accum += log_probs[idx];
    }
    cum_log_probs[bidx] = accum;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <max_length> <batch_size> <vocab_size> <repeat>\n", argv[0]);
    return 1;
  }
  const int max_length        = atoi(argv[1]);
  const int batch_size        = atoi(argv[2]);
  const int vocab_size        = atoi(argv[3]);
  const int repeat            = atoi(argv[4]);
  const int vocab_size_padded = vocab_size;

  const size_t logits_n    = (size_t)batch_size * max_length * vocab_size_padded;
  const size_t ids_n       = (size_t)batch_size * max_length;
  const size_t log_probs_n = (size_t)batch_size * (max_length - 1);

  std::vector<float> h_logits(logits_n);
  {
    std::default_random_engine rng(123);
    std::uniform_real_distribution<float> ud(-6.f, 6.f);
    for (auto& v : h_logits) v = ud(rng);
  }

  std::vector<int>   h_lengths(batch_size, max_length);
  std::vector<int>   h_ids(ids_n);
  srand(123);
  for (auto& v : h_ids) v = rand() % vocab_size;

  std::vector<float> h_log_probs_ref(log_probs_n);
  std::vector<float> h_cum_ref(batch_size);
  log_probs_cpu(h_log_probs_ref.data(), h_logits.data(), h_ids.data(),
                h_lengths.data(), max_length, batch_size,
                vocab_size, vocab_size_padded, h_cum_ref.data());

  float* d_logits    = (float*)malloc(logits_n * sizeof(float));
  float* d_log_probs = (float*)malloc(log_probs_n * sizeof(float));
  float* d_cum       = (float*)malloc(batch_size * sizeof(float));
  int*   d_ids       = (int*)  malloc(ids_n * sizeof(int));
  int*   d_lengths   = (int*)  malloc(batch_size * sizeof(int));

  memcpy(d_logits,  h_logits.data(),  logits_n * sizeof(float));
  memcpy(d_ids,     h_ids.data(),     ids_n * sizeof(int));
  memcpy(d_lengths, h_lengths.data(), batch_size * sizeof(int));

  #pragma omp target enter data \
      map(to: d_logits[0:logits_n], d_ids[0:ids_n], d_lengths[0:batch_size]) \
      map(alloc: d_log_probs[0:log_probs_n], d_cum[0:batch_size])

  auto run_kernels = [&]() {
    const int num_teams1 = (max_length - 1) * batch_size;

    // Kernel 1: log_probs per (step, batch) -- each team handles one (step, bidx)
    #pragma omp target teams num_teams(num_teams1) thread_limit(256) \
        map(tofrom: d_log_probs[0:log_probs_n])
    {
      int league_rank = omp_get_team_num();
      int step  = league_rank % (max_length - 1);
      int bidx  = league_rank / (max_length - 1);

      if (bidx < batch_size && step < d_lengths[bidx] - 1) {
        int base = bidx * max_length * vocab_size_padded + step * vocab_size_padded;

        float max_val = -FLT_MAX;
        #pragma omp parallel for reduction(max: max_val)
        for (int i = 0; i < vocab_size; i++)
          max_val = fmaxf(max_val, d_logits[base + i]);

        float sum_exp = 0.f;
        #pragma omp parallel for reduction(+: sum_exp)
        for (int i = 0; i < vocab_size; i++)
          sum_exp += expf(d_logits[base + i] - max_val);

        #pragma omp single
        {
          int idx       = step + bidx * (max_length - 1);
          int token_idx = step + 1 + bidx * max_length;
          d_log_probs[idx] = d_logits[base + d_ids[token_idx]]
                             - max_val - logf(sum_exp + 1e-9f);
        }
      }
    }

    // Kernel 2: accumulate log_probs per batch entry
    #pragma omp target teams num_teams(batch_size) thread_limit(256) \
        map(tofrom: d_cum[0:batch_size])
    {
      int bidx   = omp_get_team_num();
      int length = d_lengths[bidx];
      int base   = bidx * (max_length - 1);

      float accum = 0.f;
      #pragma omp parallel for reduction(+: accum)
      for (int step = 0; step < length - 1; step++)
        accum += d_log_probs[base + step];

      #pragma omp single
      d_cum[bidx] = accum;
    }
  };

  // Warmup + verify
  run_kernels();

  #pragma omp target update from(d_log_probs[0:log_probs_n], d_cum[0:batch_size])

  bool error = false;
  for (size_t i = 0; i < log_probs_n && !error; ++i) {
    if (fabsf(d_log_probs[i] - h_log_probs_ref[i]) > 1e-3f) {
      printf("log_probs mismatch @%zu: %f vs %f\n", i, d_log_probs[i], h_log_probs_ref[i]);
      error = true;
    }
  }
  for (int i = 0; i < batch_size && !error; ++i) {
    if (fabsf(d_cum[i] - h_cum_ref[i]) > 1e-1f) {
      printf("cum_log_probs mismatch @%d: %f vs %f\n", i, d_cum[i], h_cum_ref[i]);
      error = true;
    }
  }
  printf("%s\n", error ? "FAIL" : "PASS");

  // Timed run
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; ++r) run_kernels();
  auto t1 = std::chrono::steady_clock::now();
  printf("Average execution time of kernels: %f (us)\n",
         std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
             * 1e-3 / repeat);

  #pragma omp target exit data \
      map(delete: d_logits[0:logits_n], d_ids[0:ids_n], d_lengths[0:batch_size], \
                  d_log_probs[0:log_probs_n], d_cum[0:batch_size])

  free(d_logits); free(d_log_probs); free(d_cum);
  free(d_ids); free(d_lengths);
  return 0;
}
