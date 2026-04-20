// Single-rank AllReduce bandwidth benchmark – OpenMP target offloading port
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

int main(int argc, char* argv[]) {
  int repeat = 5;
  if (argc > 1) repeat = std::atoi(argv[1]);

  const int nRanks = 1;

  const size_t sizes[] = {
    1024ULL * 1024,
    10ULL * 1024 * 1024,
    100ULL * 1024 * 1024,
    1000ULL * 1024 * 1024
  };

  printf("%-20s  %-25s  %-15s\n",
         "Transfer Size (B)", "Avg Transfer Time (s)", "Bandwidth (GB/s)");

  for (size_t size : sizes) {
    const size_t num_B  = sizeof(float) * size * nRanks;
    const double num_GB = (double)num_B / 1e9;

    float* sendbuff = (float*)malloc(size * sizeof(float));
    float* recvbuff = (float*)malloc(size * sizeof(float));

    #pragma omp target enter data map(alloc: sendbuff[0:size], recvbuff[0:size])

    // Initialise sendbuff to 1.0f
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (size_t i = 0; i < size; i++) sendbuff[i] = 1.0f;

    auto t0 = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; ++r) {
      double total = 0.0;
      #pragma omp target teams distribute parallel for reduction(+:total) thread_limit(256)
      for (size_t i = 0; i < size; i++) total += (double)sendbuff[i];

      float reduced_val = (float)(total / (double)size);
      #pragma omp target teams distribute parallel for thread_limit(256)
      for (size_t i = 0; i < size; i++) recvbuff[i] = reduced_val;
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed  = std::chrono::duration<double>(t1 - t0).count();
    double avg_time = elapsed / repeat;
    double bandwidth= num_GB / avg_time;

    // Verify
    float h_val;
    #pragma omp target update from(recvbuff[0:1])
    h_val = recvbuff[0];
    if (std::fabs(h_val - (float)nRanks) > 1e-4f)
      printf("VERIFICATION FAILED at size=%zu: got %.6f expected %.6f\n",
             size, (double)h_val, (double)nRanks);

    printf("%-20zu  %-25.6f  %-15.4f\n", num_B, avg_time, bandwidth);

    #pragma omp target exit data map(delete: sendbuff[0:size], recvbuff[0:size])
    free(sendbuff);
    free(recvbuff);
  }
  return 0;
}
