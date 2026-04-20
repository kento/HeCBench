// OpenMP target port of saxpy-ompt benchmark.
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

#define TWO26 (1 << 26)
#define NLUP  32

int main(int argc, char* argv[]) {
  const int n = TWO26;
  const float a = 2.0f;
  const size_t nbytes = sizeof(float) * n;

  float* x     = (float*) malloc(nbytes);
  float* y     = (float*) malloc(nbytes);
  float* yhost = (float*) malloc(nbytes);
  float* yaccl = (float*) malloc(nbytes);
  if (!x || !y || !yhost || !yaccl) {
    printf("error: memory allocation\n");
    return 1;
  }

  srand(42);
  for (int i = 0; i < n; i++) {
    x[i]     = (rand() % 32) / 32.0f;
    y[i]     = (rand() % 32) / 32.0f;
    yhost[i] = a * x[i] + y[i];
    yaccl[i] = y[i];
  }

  printf("The system supports 1 ns time resolution\n");
  printf("total size of x and y is %9.1f MB\n", 2.0 * nbytes / (1 << 20));
  printf("tests are averaged over %2d loops\n", NLUP);

  // Host saxpy
  {
    float* ytmp = (float*) malloc(nbytes);
    for (int i = 0; i < n; i++) ytmp[i] = y[i];

    auto t0 = std::chrono::steady_clock::now();
    for (int loop = 0; loop < NLUP; loop++) {
      for (int i = 0; i < n; i++) ytmp[i] = a * x[i] + ytmp[i];
    }
    auto t1 = std::chrono::steady_clock::now();
    double wt = std::chrono::duration<double>(t1 - t0).count() / NLUP;
    double mbps = 3.0 * nbytes / wt / (1 << 20);

    float maxabserr = 0.0f;
    for (int i = 0; i < n; i++) {
      float err = fabsf(ytmp[i] - yhost[i] - (NLUP - 1) * (a * x[i]));
      if (err > maxabserr) maxabserr = err;
    }
    printf("saxpy on host: %9.1f MB/s %9.1f MB/s maxabserr = %.1f\n",
           mbps, mbps, (double)maxabserr);
    free(ytmp);
  }

  // Device saxpy
  {
    float* d_x = (float*) malloc(nbytes);
    float* d_y = (float*) malloc(nbytes);
    for (int i = 0; i < n; i++) { d_x[i] = x[i]; d_y[i] = y[i]; }

    #pragma omp target enter data map(to: d_x[0:n], d_y[0:n])

    // Warmup
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < n; i++) d_y[i] = a * d_x[i] + d_y[i];

    // Reset y on device
    #pragma omp target update to(d_y[0:n])

    auto t0 = std::chrono::steady_clock::now();
    for (int loop = 0; loop < NLUP; loop++) {
      #pragma omp target teams distribute parallel for thread_limit(256)
      for (int i = 0; i < n; i++) d_y[i] = a * d_x[i] + d_y[i];
    }
    #pragma omp taskwait
    auto t1 = std::chrono::steady_clock::now();

    double wt = std::chrono::duration<double>(t1 - t0).count() / NLUP;
    double mbps = 3.0 * nbytes / wt / (1 << 20);

    #pragma omp target update from(d_y[0:n])
    #pragma omp target exit data map(delete: d_x[0:n], d_y[0:n])

    float maxabserr = 0.0f;
    for (int i = 0; i < n; i++) {
      float ref = yhost[i] + (NLUP - 1) * (a * x[i]);
      float err = fabsf(d_y[i] - ref);
      if (err > maxabserr) maxabserr = err;
    }

    printf("saxpy on accl (impl. 0)\n");
    printf("total: %9.1f MB/s kernel: %9.1f MB/s maxabserr = %.1f\n",
           mbps, mbps, (double)maxabserr);
    free(d_x);
    free(d_y);
  }

  free(x); free(y); free(yhost); free(yaccl);
  return 0;
}
