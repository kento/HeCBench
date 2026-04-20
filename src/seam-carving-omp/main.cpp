// OpenMP target port of seam-carving benchmark.
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <chrono>
#include <algorithm>
#include <vector>

static inline int make_pixel(int r, int g, int b)
{
  return (r & 0xFF) | ((g & 0xFF) << 8) | ((b & 0xFF) << 16);
}
static inline int pixel_r(int p) { return  p        & 0xFF; }
static inline int pixel_g(int p) { return (p >>  8) & 0xFF; }
static inline int pixel_b(int p) { return (p >> 16) & 0xFF; }

int main(int argc, char *argv[])
{
  if (argc != 5) {
    printf("Usage: %s <width> <height> <seams_to_remove> <repeat>\n", argv[0]);
    return 1;
  }

  int width           = atoi(argv[1]);
  int height          = atoi(argv[2]);
  int seams_to_remove = atoi(argv[3]);
  int repeat          = atoi(argv[4]);

  if (seams_to_remove < 0 || seams_to_remove >= width) {
    printf("ERROR: seams_to_remove must be in [0, width).\n");
    return 1;
  }

  printf("Image size: %dx%d\n", width, height);
  printf("Removing %d seams...\n", seams_to_remove);

  int total_pixels = width * height;
  std::vector<int> h_pixels(total_pixels);
  for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++) {
      int r = (x * 37 + y * 53) % 256;
      int g = (x * 61 + y * 29) % 256;
      int bv = (x * 13 + y * 97) % 256;
      h_pixels[y * width + x] = make_pixel(r, g, bv);
    }

  int* d_pixels = (int*) malloc((size_t)width * height * sizeof(int));
  int* d_energy = (int*) malloc((size_t)width * height * sizeof(int));
  int* d_dp     = (int*) malloc((size_t)width * height * sizeof(int));
  int* d_seam   = (int*) malloc(height * sizeof(int));
  int* d_out    = (int*) malloc((size_t)width * height * sizeof(int));
  int* h_dp     = (int*) malloc((size_t)width * height * sizeof(int));
  int* h_seam   = (int*) malloc(height * sizeof(int));

  size_t buf_size = (size_t)width * height;

  #pragma omp target enter data map(alloc: d_pixels[0:buf_size], d_energy[0:buf_size], \
                                    d_dp[0:buf_size], d_seam[0:height], d_out[0:buf_size])

  double total_time_us = 0.0;

  for (int run = 0; run < repeat; run++) {
    int cur_w = width;
    // Reset pixels
    for (size_t i = 0; i < buf_size; i++) d_pixels[i] = h_pixels[i];
    #pragma omp target update to(d_pixels[0:buf_size])

    auto t0 = std::chrono::steady_clock::now();

    for (int s = 0; s < seams_to_remove; s++) {
      const int W = cur_w;
      const int H = height;

      // Energy computation
      #pragma omp target teams distribute parallel for thread_limit(256)
      for (int idx = 0; idx < W * H; idx++) {
        int x = idx % W;
        int y = idx / W;

        int xl = (x > 0)     ? x - 1 : x;
        int xr = (x < W - 1) ? x + 1 : x;
        int yu = (y > 0)     ? y - 1 : y;
        int yd = (y < H - 1) ? y + 1 : y;

        int pl = d_pixels[y  * W + xl];
        int pr = d_pixels[y  * W + xr];
        int pu = d_pixels[yu * W + x];
        int pd = d_pixels[yd * W + x];

        int e = 0;
        e += abs(pixel_r(pr) - pixel_r(pl)) + abs(pixel_g(pr) - pixel_g(pl)) + abs(pixel_b(pr) - pixel_b(pl));
        e += abs(pixel_r(pd) - pixel_r(pu)) + abs(pixel_g(pd) - pixel_g(pu)) + abs(pixel_b(pd) - pixel_b(pu));
        d_energy[idx] = e;
      }

      // DP: copy first row
      #pragma omp target teams distribute parallel for thread_limit(256)
      for (int x = 0; x < W; x++)
        d_dp[x] = d_energy[x];

      // DP: row by row
      for (int y = 1; y < H; y++) {
        const int row = y;
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int x = 0; x < W; x++) {
          int best = d_dp[(row - 1) * W + x];
          if (x > 0     && d_dp[(row - 1) * W + x - 1] < best) best = d_dp[(row - 1) * W + x - 1];
          if (x < W - 1 && d_dp[(row - 1) * W + x + 1] < best) best = d_dp[(row - 1) * W + x + 1];
          d_dp[row * W + x] = d_energy[row * W + x] + best;
        }
      }

      // Seam finding on host
      #pragma omp target update from(d_dp[0:W*H])
      int min_idx = 0;
      for (int x = 1; x < W; x++)
        if (d_dp[(H - 1) * W + x] < d_dp[(H - 1) * W + min_idx])
          min_idx = x;
      h_seam[H - 1] = min_idx;
      for (int y = H - 2; y >= 0; y--) {
        int cx = h_seam[y + 1];
        int best = cx;
        if (cx > 0     && d_dp[y * W + cx - 1] < d_dp[y * W + best]) best = cx - 1;
        if (cx < W - 1 && d_dp[y * W + cx + 1] < d_dp[y * W + best]) best = cx + 1;
        h_seam[y] = best;
      }
      for (int i = 0; i < H; i++) d_seam[i] = h_seam[i];
      #pragma omp target update to(d_seam[0:H])

      // Seam removal
      const int newW = W - 1;
      #pragma omp target teams distribute parallel for thread_limit(256)
      for (int y = 0; y < H; y++) {
        int sc = d_seam[y];
        for (int x = 0; x < sc; x++)
          d_out[y * newW + x] = d_pixels[y * W + x];
        for (int x = sc; x < newW; x++)
          d_out[y * newW + x] = d_pixels[y * W + x + 1];
      }

      // Swap: copy d_out -> d_pixels
      #pragma omp target teams distribute parallel for thread_limit(256)
      for (int i = 0; i < newW * H; i++)
        d_pixels[i] = d_out[i];

      cur_w = newW;
    }

    auto t1 = std::chrono::steady_clock::now();
    total_time_us += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-3;
  }

  #pragma omp target exit data map(delete: d_pixels[0:buf_size], d_energy[0:buf_size], \
                                   d_dp[0:buf_size], d_seam[0:height], d_out[0:buf_size])

  double avg_us = total_time_us / repeat / seams_to_remove;
  printf("Average time per seam: %f (us)\n", avg_us);
  printf("Done. Final width: %d\n", width - seams_to_remove);

  free(d_pixels); free(d_energy); free(d_dp);
  free(d_seam); free(d_out); free(h_dp); free(h_seam);
  return 0;
}
