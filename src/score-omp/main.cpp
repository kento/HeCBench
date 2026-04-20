// OpenMP target port of score benchmark (object detection top-K scoring).
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const float threshold = 0.4f;
  const int classwise_topK = 10;
  const int num_classes = 1000;
  const int num_priors = 1024;
  const int batch_size = 512;
  const int BINS = 2048;

  const size_t scores_size  = (size_t)batch_size * num_classes * num_priors;
  const size_t indices_size = (size_t)batch_size * num_classes * classwise_topK;
  const size_t count_size   = (size_t)batch_size * num_classes;

  float* scores  = (float*) malloc(scores_size  * sizeof(float));
  int*   count   = (int*)   malloc(count_size   * sizeof(int));
  int*   indices = (int*)   malloc(indices_size * sizeof(int));

  srand(123);
  for (size_t i = 0; i < scores_size; i++)
    scores[i] = rand() / (float)RAND_MAX;

  memset(count,   0, count_size   * sizeof(int));
  memset(indices, 0, indices_size * sizeof(int));

  #pragma omp target enter data map(to: scores[0:scores_size]) \
                                map(alloc: count[0:count_size], indices[0:indices_size])

  const int nc   = num_classes;
  const int np   = num_priors;
  const int topK = classwise_topK;
  const float thr = threshold;
  const int bins_count = BINS;

  auto t0 = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    // Zero count and indices before each iteration
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (size_t i = 0; i < count_size; i++) count[i] = 0;

    #pragma omp target teams distribute parallel for thread_limit(256)
    for (size_t i = 0; i < indices_size; i++) indices[i] = 0;

    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int bc = 0; bc < batch_size * nc; bc++) {
      const int b = bc / nc;
      const int c = bc % nc;

      const float* sc = scores + (b * nc + c) * np;
      int* cnt = count   + b * nc + c;
      int* idx = indices + (b * nc + c) * topK;

      int bins[2048];
      for (int k = 0; k < bins_count; k++) bins[k] = 0;

      for (int i = 0; i < np; i++) {
        float confidence = sc[i];
        if (confidence > thr) {
          float conf_scaled = (confidence - thr) / (1.0f - thr);
          int bin_index = (int)(conf_scaled * bins_count);
          if (bin_index >= bins_count) bin_index = bins_count - 1;
          bin_index -= 1;
          if (bin_index >= 0)
            bins[bin_index]++;
        }
      }

      for (int i = bins_count - 2; i >= 0; i--)
        bins[i] += bins[i + 1];

      int total = 0;
      for (int i = 0; i < np; i++) {
        if (sc[i] > thr) total++;
      }
      *cnt = (total < topK) ? total : topK;

      int written = 0;
      for (int i = 0; i < np && written < *cnt; i++) {
        float confidence = sc[i];
        if (confidence > thr) {
          float conf_scaled = (confidence - thr) / (1.0f - thr);
          int bin_index = (int)(conf_scaled * bins_count);
          if (bin_index >= bins_count) bin_index = bins_count - 1;
          int pos = bins_count - 1 - bin_index;
          if (pos < topK) {
            idx[written++] = i;
          }
        }
      }
    }
  }

  auto t1 = std::chrono::steady_clock::now();
  double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
  printf("Average kernel execution time: %f (ms)\n", ns * 1e-6 / repeat);

  #pragma omp target update from(count[0:count_size])
  #pragma omp target exit data map(delete: scores[0:scores_size], \
                                   count[0:count_size], indices[0:indices_size])

  long checksum = 0;
  for (size_t i = 0; i < count_size; i++) checksum += count[i];
  printf("Checksum (count) = %ld\n", checksum);

  free(scores); free(count); free(indices);
  return 0;
}
