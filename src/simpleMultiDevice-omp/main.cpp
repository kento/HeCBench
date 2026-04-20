// simpleMultiDevice OpenMP target port (single device)
// Original: distributes float reduction across multiple GPUs.
// OMP port: single-device parallel reduction over DATA_N floats.

#include <omp.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>

static const int DATA_N = 1048576 * 32;  // 33554432

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  std::vector<float> h_data(DATA_N);
  srand(42);
  for (int i = 0; i < DATA_N; i++)
    h_data[i] = (float)rand() / (float)RAND_MAX;

  printf("Starting simpleMultiDevice\n");
  printf("GPU device count: 1\n");
  printf("Generating input data of size %d ...\n\n", DATA_N);

  double *d_data = (double*)malloc(DATA_N * sizeof(double));
  for (int i = 0; i < DATA_N; i++) d_data[i] = (double)h_data[i];

  #pragma omp target enter data map(to: d_data[0:DATA_N])

  printf("Computing with 1 GPUs...\n");

  double gpu_sum_d = 0.0;
  auto t0 = std::chrono::steady_clock::now();
  for (int k = 0; k < repeat; k++) {
    double local_sum = 0.0;
    #pragma omp target teams distribute parallel for reduction(+:local_sum) thread_limit(256)
    for (int i = 0; i < DATA_N; i++) local_sum += d_data[i];
    gpu_sum_d = local_sum;
  }
  auto t1 = std::chrono::steady_clock::now();
  double elapsed_us =
    std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
    * 1e-3 / repeat;
  printf("  Average GPU Processing time: %f (us)\n\n", elapsed_us);

  float gpu_sum = (float)gpu_sum_d;

  #pragma omp target exit data map(delete: d_data[0:DATA_N])
  free(d_data);

  printf("Computing with Host CPU...\n\n");
  double cpu_sum = 0.0;
  for (int i = 0; i < DATA_N; i++) cpu_sum += (double)h_data[i];

  printf("Comparing GPU and Host CPU results...\n");
  double diff = fabs(cpu_sum - (double)gpu_sum) / fabs(cpu_sum);
  printf("  GPU sum: %f\n  CPU sum: %f\n", gpu_sum, (float)cpu_sum);
  printf("  Relative difference: %E \n\n", diff);

  return (diff < 1e-5) ? EXIT_SUCCESS : EXIT_FAILURE;
}
