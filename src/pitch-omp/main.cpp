#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

static int padded_width(int w) {
  return ((w * (int)sizeof(float) + 63) & ~63) / (int)sizeof(float);
}

void malloc2D(int repeat, int width, int height) {
  printf("Dimension: (%d %d)\n", width, height);
  const int wp = padded_width(width);
  const size_t pitched_size = (size_t)wp * height;
  const size_t simple_size  = (size_t)width * height;

  float* d_pitched = (float*)malloc(pitched_size * sizeof(float));
  float* d_simple  = (float*)malloc(simple_size  * sizeof(float));

  srand(42);
  for (int r = 0; r < height; r++) {
    for (int c = 0; c < width; c++) {
      float val = (float)rand() / (float)RAND_MAX;
      d_pitched[r * wp    + c] = val;
      d_simple [r * width + c] = val;
    }
  }
#pragma omp target enter data map(to: d_pitched[0:pitched_size], d_simple[0:simple_size])

  // Pitched warm-up
#pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
  for (int r = 0; r < height; r++) {
    for (int c = 0; c < width; c++) {
      int idx = r * wp + c;
      d_pitched[idx] = 1.0f / (1.0f + expf(-d_pitched[idx]));
    }
  }

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
#pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
    for (int r = 0; r < height; r++) {
      for (int c = 0; c < width; c++) {
        int idx = r * wp + c;
        d_pitched[idx] = 1.0f / (1.0f + expf(-d_pitched[idx]));
      }
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  double time_pitched = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-3;

  // Simple warm-up
#pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
  for (int r = 0; r < height; r++) {
    for (int c = 0; c < width; c++) {
      int idx = r * width + c;
      d_simple[idx] = 1.0f / (1.0f + expf(-d_simple[idx]));
    }
  }

  auto t2 = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
#pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
    for (int r = 0; r < height; r++) {
      for (int c = 0; c < width; c++) {
        int idx = r * width + c;
        d_simple[idx] = 1.0f / (1.0f + expf(-d_simple[idx]));
      }
    }
  }
  auto t3 = std::chrono::steady_clock::now();
  double time_simple = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count() * 1e-3;

  printf("Average execution time (pitched vs simple): %f %f (us)\n",
         (float)(time_pitched / repeat), (float)(time_simple / repeat));

#pragma omp target exit data map(delete: d_pitched[0:pitched_size], d_simple[0:simple_size])
  free(d_pitched);
  free(d_simple);
}

void malloc3D(int repeat, int width, int height, int depth) {
  printf("Dimension: (%d %d %d)\n", width, height, depth);
  const int wp = padded_width(width);
  const size_t pitched_size = (size_t)wp    * height * depth;
  const size_t simple_size  = (size_t)width * height * depth;

  float* d_pitched = (float*)malloc(pitched_size * sizeof(float));
  float* d_simple  = (float*)malloc(simple_size  * sizeof(float));

  srand(42);
  for (int z = 0; z < depth;  z++)
  for (int y = 0; y < height; y++)
  for (int x = 0; x < width;  x++) {
    float val = (float)rand() / (float)RAND_MAX;
    d_pitched[z * height * wp    + y * wp    + x] = val;
    d_simple [z * height * width + y * width + x] = val;
  }
#pragma omp target enter data map(to: d_pitched[0:pitched_size], d_simple[0:simple_size])

  // Pitched warm-up
#pragma omp target teams distribute parallel for collapse(3) thread_limit(256)
  for (int z = 0; z < depth;  z++) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = z * height * wp + y * wp + x;
        d_pitched[idx] = 1.0f / (1.0f + expf(-d_pitched[idx]));
      }
    }
  }

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
#pragma omp target teams distribute parallel for collapse(3) thread_limit(256)
    for (int z = 0; z < depth;  z++) {
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          int idx = z * height * wp + y * wp + x;
          d_pitched[idx] = 1.0f / (1.0f + expf(-d_pitched[idx]));
        }
      }
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  double time_pitched = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-3;

  // Simple warm-up
#pragma omp target teams distribute parallel for collapse(3) thread_limit(256)
  for (int z = 0; z < depth;  z++) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = z * height * width + y * width + x;
        d_simple[idx] = 1.0f / (1.0f + expf(-d_simple[idx]));
      }
    }
  }

  auto t2 = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
#pragma omp target teams distribute parallel for collapse(3) thread_limit(256)
    for (int z = 0; z < depth;  z++) {
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          int idx = z * height * width + y * width + x;
          d_simple[idx] = 1.0f / (1.0f + expf(-d_simple[idx]));
        }
      }
    }
  }
  auto t3 = std::chrono::steady_clock::now();
  double time_simple = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count() * 1e-3;

  printf("Average execution time (pitched vs simple): %f %f (us)\n",
         (float)(time_pitched / repeat), (float)(time_simple / repeat));

#pragma omp target exit data map(delete: d_pitched[0:pitched_size], d_simple[0:simple_size])
  free(d_pitched);
  free(d_simple);
}

int main(int argc, char* argv[]) {
  if (argc != 2) { printf("Usage: %s <repeat>\n", argv[0]); return 1; }
  const int repeat = atoi(argv[1]);

  const int w[] = {227,256,720,768,854,1280,1440,1920,2048,3840,4096};
  const int h[] = {227,256,480,576,480,720,1080,1080,1080,2160,2160};
  const int d[] = {1,3};
  for (int i = 0; i < 11; i++) malloc2D(repeat, w[i], h[i]);
  for (int i = 0; i < 11; i++)
    for (int j = 0; j < 2; j++)
      malloc3D(repeat, w[i], h[i], d[j]);
  return 0;
}
