// OpenMP target port of matrixT-kokkos: matrix transpose with 8 variants.

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

#define TILE_DIM   16
#define BLOCK_ROWS 16

static void computeTransposeGold(float* gold, const float* idata,
                                  int size_x, int size_y) {
  for (int y = 0; y < size_y; ++y)
    for (int x = 0; x < size_x; ++x)
      gold[(x * size_y) + y] = idata[(y * size_x) + x];
}

int main(int argc, char** argv) {
  if (argc != 4) {
    printf("Usage: %s <row_dim> <col_dim> <repeat>\n", argv[0]);
    return 0;
  }

  int size_x = atoi(argv[1]);
  int size_y = atoi(argv[2]);
  int repeat  = atoi(argv[3]);

  if (size_x != size_y) {
    printf("Error: non-square matrices (row_dim_size(%d) != col_dim_size(%d))\nExiting...\n\n",
           size_x, size_y);
    return 1;
  }
  if (size_x % TILE_DIM != 0 || size_y % TILE_DIM != 0) {
    printf("Matrix size must be integral multiple of tile size\nExiting...\n\n");
    return 1;
  }

  const int mat_size = size_x * size_y;

  float* d_idata = (float*)malloc(mat_size * sizeof(float));
  float* d_odata = (float*)malloc(mat_size * sizeof(float));

  for (int i = 0; i < mat_size; i++)
    d_idata[i] = (float)i;

  #pragma omp target enter data \
      map(to: d_idata[0:mat_size]) \
      map(alloc: d_odata[0:mat_size])

  float* h_idata = d_idata;
  float* transposeGold = new float[mat_size];
  computeTransposeGold(transposeGold, h_idata, size_x, size_y);

  printf("\nMatrix size: %dx%d (%dx%d tiles), tile size: %dx%d, block size: %dx%d\n\n",
         size_x, size_y, size_x/TILE_DIM, size_y/TILE_DIM,
         TILE_DIM, TILE_DIM, TILE_DIM, BLOCK_ROWS);

  const int TD         = TILE_DIM;
  const int sx         = size_x;
  const int sy         = size_y;
  const int gridDim_x  = size_x / TILE_DIM;

  bool success = true;

  for (int k = 0; k < 8; k++) {
    const char* kernelName = nullptr;

    auto t0 = std::chrono::steady_clock::now();

    switch (k) {
      case 0: {
        kernelName = "simple copy       ";
        for (int iter = 0; iter < repeat; iter++) {
          #pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
          for (int y = 0; y < sy; y++)
            for (int x = 0; x < sx; x++)
              d_odata[y * sx + x] = d_idata[y * sx + x];
        }
        break;
      }
      case 1: {
        kernelName = "shared memory copy";
        for (int iter = 0; iter < repeat; iter++) {
          #pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
          for (int y = 0; y < sy; y++)
            for (int x = 0; x < sx; x++)
              d_odata[y * sx + x] = d_idata[y * sx + x];
        }
        break;
      }
      case 2: {
        kernelName = "naive             ";
        for (int iter = 0; iter < repeat; iter++) {
          #pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
          for (int y = 0; y < sy; y++)
            for (int x = 0; x < sx; x++)
              d_odata[y + sy * x] = d_idata[x + sx * y];
        }
        break;
      }
      case 3: {
        kernelName = "coalesced         ";
        for (int iter = 0; iter < repeat; iter++) {
          #pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
          for (int y = 0; y < sy; y++) {
            for (int x = 0; x < sx; x++) {
              const int bx = x / TD, tx = x % TD;
              const int by = y / TD, ty = y % TD;
              const int index_out = (by * TD + tx) + (bx * TD + ty) * sy;
              const int index_in  = (bx * TD + ty) + (by * TD + tx) * sx;
              d_odata[index_out] = d_idata[index_in];
            }
          }
        }
        break;
      }
      case 4: {
        kernelName = "optimized         ";
        for (int iter = 0; iter < repeat; iter++) {
          #pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
          for (int y = 0; y < sy; y++) {
            for (int x = 0; x < sx; x++) {
              const int bx = x / TD, tx = x % TD;
              const int by = y / TD, ty = y % TD;
              const int index_out = (by * TD + tx) + (bx * TD + ty) * sy;
              const int index_in  = (bx * TD + ty) + (by * TD + tx) * sx;
              d_odata[index_out] = d_idata[index_in];
            }
          }
        }
        break;
      }
      case 5: {
        kernelName = "coarse-grained    ";
        for (int iter = 0; iter < repeat; iter++) {
          #pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
          for (int y = 0; y < sy; y++) {
            for (int x = 0; x < sx; x++) {
              const int bx = x / TD, tx = x % TD;
              const int by = y / TD, ty = y % TD;
              const int index_in  = x + y * sx;
              const int index_out = (by * TD + tx) + (bx * TD + ty) * sy;
              d_odata[index_out] = d_idata[index_in];
            }
          }
        }
        break;
      }
      case 6: {
        kernelName = "fine-grained      ";
        for (int iter = 0; iter < repeat; iter++) {
          #pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
          for (int y = 0; y < sy; y++) {
            for (int x = 0; x < sx; x++) {
              const int bx = x / TD, tx = x % TD;
              const int by = y / TD, ty = y % TD;
              const int index_out = x + y * sx;
              const int index_in  = (bx * TD + ty) + (by * TD + tx) * sx;
              d_odata[index_out] = d_idata[index_in];
            }
          }
        }
        break;
      }
      case 7: {
        kernelName = "diagonal          ";
        const int nblocks_x = sx / TD;
        const int nblocks_y = sy / TD;
        for (int iter = 0; iter < repeat; iter++) {
          #pragma omp target teams distribute parallel for collapse(2) thread_limit(256) \
              firstprivate(nblocks_x, nblocks_y)
          for (int y = 0; y < sy; y++) {
            for (int x = 0; x < sx; x++) {
              const int bx_raw = x / TD, tx = x % TD;
              const int by_raw = y / TD, ty = y % TD;

              int blockIdx_x, blockIdx_y;
              if (sx == sy) {
                blockIdx_y = bx_raw;
                blockIdx_x = (bx_raw + by_raw) % gridDim_x;
              } else {
                const int b = bx_raw + nblocks_x * by_raw;
                blockIdx_y  = b % nblocks_y;
                blockIdx_x  = ((b / nblocks_y) + blockIdx_y) % nblocks_x;
              }

              const int index_out = (blockIdx_y * TD + tx) + (blockIdx_x * TD + ty) * sy;
              const int index_in  = (blockIdx_x * TD + ty) + (blockIdx_y * TD + tx) * sx;
              d_odata[index_out] = d_idata[index_in];
            }
          }
        }
        break;
      }
    }

    auto t1 = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    printf("Average kernel (%s) execution time: %f (us)\n",
           kernelName, (elapsed * 1e-3f) / repeat);

    #pragma omp target update from(d_odata[0:mat_size])

    const float* gold = nullptr;
    if (k == 0 || k == 1) {
      gold = h_idata;
    } else if (k == 5 || k == 6) {
      gold = d_odata;
    } else {
      gold = transposeGold;
    }

    bool ok = true;
    for (int i = 0; i < mat_size; i++) {
      if (std::fabs(gold[i] - d_odata[i]) > 0.01f) {
        ok = false;
        break;
      }
    }
    if (!ok) {
      printf("*** case %d: kernel FAILED ***\n", k);
      success = false;
    }
  }

  printf("%s\n", success ? "PASS" : "FAIL");

  #pragma omp target exit data \
      map(delete: d_idata[0:mat_size], d_odata[0:mat_size])

  free(d_idata); free(d_odata);
  delete[] transposeGold;
  return 0;
}
