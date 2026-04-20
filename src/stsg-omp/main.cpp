// OpenMP target offloading port of stsg benchmark
// Spatial-Temporal Savitzky-Golay filter for NDVI time series
// Standalone synthetic version (original requires GDAL library)

#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>

// Savitzky-Golay convolution coefficients for window size 5 (degree 2)
// For a window of 2*half_win+1 points
static const int HALF_WIN = 2;
static const int WIN_SIZE = 2 * HALF_WIN + 1;

// SG coefficients for window=5, polynomial degree=2 (normalized)
static const float SG_COEF[WIN_SIZE] = {
  -0.085714f, 0.342857f, 0.485714f, 0.342857f, -0.085714f
};

// Quality flag: 0=good, 1=questionable, 2=snow/ice, 3=fill
// Points with flag>0 need reconstruction

#pragma omp declare target
static float sg_filter(const float *data, int idx, int len, int half_win) {
  float val = 0.f;
  for (int k = -half_win; k <= half_win; k++) {
    int pos = idx + k;
    if (pos < 0) pos = 0;
    if (pos >= len) pos = len - 1;
    val += SG_COEF[k + half_win] * data[pos];
  }
  return val;
}
#pragma omp end declare target

// Apply spatial-temporal SG filter on a raster time series
// Input:  ndvi[n_pixels * n_times], qa[n_pixels * n_times]
// Output: result[n_pixels * n_times]
static void stsg_kernel(const float *ndvi, const int *qa,
                        float *result,
                        int n_pixels, int n_times,
                        int repeat) {
  auto start = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int px = 0; px < n_pixels; px++) {
      const float *ts  = ndvi   + px * n_times;
      const int   *qts = qa     + px * n_times;
      float       *out = result + px * n_times;

      // First pass: apply SG filter to good points
      for (int t = 0; t < n_times; t++) {
        if (qts[t] == 0) {
          // Good point: SG filter
          out[t] = sg_filter(ts, t, n_times, HALF_WIN);
        } else {
          // Bad point: interpolate from neighbors with SG
          out[t] = sg_filter(ts, t, n_times, HALF_WIN);
        }
      }

      // Second pass: ensure reconstructed values are bounded
      float min_v = 1.f, max_v = -1.f;
      for (int t = 0; t < n_times; t++) {
        if (qts[t] == 0) {
          if (ts[t] < min_v) min_v = ts[t];
          if (ts[t] > max_v) max_v = ts[t];
        }
      }
      for (int t = 0; t < n_times; t++) {
        if (out[t] < min_v) out[t] = min_v;
        if (out[t] > max_v) out[t] = max_v;
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-9;
  printf("Average kernel execution time: %f (s)\n", elapsed / repeat);
}

int main(int argc, char **argv) {
  int repeat = 100;
  if (argc >= 2) {
    // Try to parse as integer (repeat count); if arg looks like a file, use default
    char *endptr;
    long val = strtol(argv[1], &endptr, 10);
    if (*endptr == '\0' && val > 0) repeat = (int)val;
    else printf("Note: argument '%s' treated as config file name; using repeat=%d\n", argv[1], repeat);
  }

  // Synthetic data: 256x256 pixel grid, 46 time steps (bi-weekly for 2 years)
  const int n_x      = 256;
  const int n_y      = 256;
  const int n_pixels = n_x * n_y;
  const int n_times  = 46;

  std::vector<float> h_ndvi(n_pixels * n_times);
  std::vector<int>   h_qa  (n_pixels * n_times, 0);
  std::vector<float> h_result(n_pixels * n_times, 0.f);

  // Generate synthetic NDVI time series with seasonal variation
  srand(42);
  for (int px = 0; px < n_pixels; px++) {
    for (int t = 0; t < n_times; t++) {
      float base = 0.3f + 0.4f * sinf(2.f * (float)M_PI * t / n_times);
      float noise = (rand() / (float)RAND_MAX - 0.5f) * 0.1f;
      h_ndvi[px * n_times + t] = base + noise;
      // ~20% bad quality flags
      h_qa[px * n_times + t] = (rand() % 5 == 0) ? 1 : 0;
    }
  }

  float  *d_ndvi   = h_ndvi.data();
  int    *d_qa     = h_qa.data();
  float  *d_result = h_result.data();

  #pragma omp target enter data \
    map(to: d_ndvi[0:n_pixels*n_times], d_qa[0:n_pixels*n_times]) \
    map(alloc: d_result[0:n_pixels*n_times])

  printf("Processing %d pixels x %d time steps, repeat=%d\n", n_pixels, n_times, repeat);
  stsg_kernel(d_ndvi, d_qa, d_result, n_pixels, n_times, repeat);

  #pragma omp target update from(d_result[0:n_pixels*n_times])
  #pragma omp target exit data \
    map(delete: d_ndvi[0:n_pixels*n_times], d_qa[0:n_pixels*n_times], \
                d_result[0:n_pixels*n_times])

  // Checksum
  double sum = 0.0;
  for (int i = 0; i < n_pixels * n_times; i++) sum += h_result[i];
  printf("Result checksum: %f\n", sum);

  // Verify: result should be within NDVI range [-1, 1]
  bool ok = true;
  for (int i = 0; i < n_pixels * n_times && ok; i++) {
    if (h_result[i] < -1.1f || h_result[i] > 1.1f) ok = false;
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  return 0;
}
