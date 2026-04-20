// OpenMP target port of sddmm-batch benchmark.
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <vector>

static void init_matrix(float *matrix, int num_rows, int num_cols, int nnz)
{
  int n = num_rows * num_cols;
  std::vector<float> d(n);
  for (int i = 0; i < n; i++) d[i] = (float)i;
  srand(123);
  for (int i = n; i > 0; i--) {
    int a = i - 1;
    int b = rand() % i;
    if (a != b) std::swap(d[a], d[b]);
  }
  for (int i = 0; i < num_rows; i++)
    for (int j = 0; j < num_cols; j++)
      matrix[i * num_cols + j] =
          (d[i * num_cols + j] >= nnz) ? 0.f : (float)(drand48() + 1.0);
}

static void init_csr(int *row_indices, float *values, int *col_indices,
                     float *matrix, int num_rows, int num_cols, int nnz)
{
  row_indices[0] = 0;
  row_indices[num_rows] = nnz;
  int tmp = 0;
  for (int i = 0; i < num_rows; i++) {
    int nnz_per_row = 0;
    for (int j = 0; j < num_cols; j++) {
      if (matrix[i * num_cols + j] != 0.f) {
        values[tmp]      = matrix[i * num_cols + j];
        col_indices[tmp] = j;
        tmp++;
        nnz_per_row++;
      }
    }
    row_indices[i + 1] = row_indices[i] + nnz_per_row;
  }
}

static void sddmm_cpu(const float *A, const float *B, const float *C_dense,
                      float *C_values, const int *C_offsets, const int *C_columns,
                      int K, int M, int N)
{
  for (int i = 0; i < M; i++) {
    for (int j = C_offsets[i]; j < C_offsets[i + 1]; j++) {
      int col = C_columns[j];
      float val = 0.f;
      for (int l = 0; l < K; l++) val += A[i * K + l] * B[l * N + col];
      C_values[j] = val;
    }
  }
}

int main(int argc, char *argv[])
{
  if (argc != 8) {
    printf("Usage: %s <num_batches> <M> <K> <N> <nnz> <repeat> <verify>\n", argv[0]);
    printf("SDDMM (A, B, C) where A: M x K, B: K x N, C: M x N (sparse CSR)\n");
    return 1;
  }

  const int b      = atoi(argv[1]);
  const int M      = atoi(argv[2]);
  const int K      = atoi(argv[3]);
  const int N      = atoi(argv[4]);
  const int nnz    = atoi(argv[5]);
  const int repeat = atoi(argv[6]);
  const int verify = atoi(argv[7]);

  const int A_size = M * K;
  const int B_size = K * N;
  const int C_size = M * N;

  std::vector<float> hA(b * A_size);
  std::vector<float> hB(b * B_size);
  std::vector<float> hC_dense(C_size);
  std::vector<float> hC_values(b * nnz, 0.f);
  std::vector<int>   hC_offsets(b * (M + 1));
  std::vector<int>   hC_columns(b * nnz);
  std::vector<float> hC_result(b * nnz, 0.f);

  for (int i = 0; i < b; i++) {
    for (int j = 0; j < A_size; j++) hA[i * A_size + j] = (float)(drand48() + 1.0);
    for (int j = 0; j < B_size; j++) hB[i * B_size + j] = (float)(drand48() + 1.0);
  }

  init_matrix(hC_dense.data(), M, N, nnz);
  {
    std::vector<float> tmp_vals(nnz);
    std::vector<int>   tmp_cols(nnz);
    std::vector<int>   tmp_rows(M + 1);
    init_csr(tmp_rows.data(), tmp_vals.data(), tmp_cols.data(),
             hC_dense.data(), M, N, nnz);
    for (int i = 0; i < b; i++) {
      std::memcpy(hC_offsets.data() + i * (M + 1), tmp_rows.data(), (M + 1) * sizeof(int));
      std::memcpy(hC_columns.data() + i * nnz,     tmp_cols.data(), nnz * sizeof(int));
    }
  }

  if (verify) {
    for (int i = 0; i < b; i++) {
      sddmm_cpu(hA.data() + i * A_size,
                hB.data() + i * B_size,
                hC_dense.data(),
                hC_result.data() + i * nnz,
                hC_offsets.data() + i * (M + 1),
                hC_columns.data() + i * nnz,
                K, M, N);
    }
  }

  float*  dA       = hA.data();
  float*  dB       = hB.data();
  float*  dValues  = hC_values.data();
  int*    dOffsets = hC_offsets.data();
  int*    dColumns = hC_columns.data();

  int bMA  = b * A_size;
  int bMB  = b * B_size;
  int bNnz = b * nnz;
  int bOff = b * (M + 1);

  #pragma omp target enter data map(to: dA[0:bMA], dB[0:bMB], dOffsets[0:bOff], dColumns[0:bNnz]) \
                                map(alloc: dValues[0:bNnz])

  const int lM   = M;
  const int lK   = K;
  const int lN   = N;
  const int lNnz = nnz;
  const int lb   = b;

  // Warmup
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int bm = 0; bm < lb * lM; bm++) {
    int batch = bm / lM;
    int row   = bm % lM;
    int row_start = dOffsets[batch * (lM + 1) + row];
    int row_end   = dOffsets[batch * (lM + 1) + row + 1];
    for (int j = row_start; j < row_end; j++) {
      int col = dColumns[batch * lNnz + j];
      float val = 0.f;
      for (int l = 0; l < lK; l++)
        val += dA[batch * lM * lK + row * lK + l] *
               dB[batch * lK * lN + l * lN + col];
      dValues[batch * lNnz + j] = val;
    }
  }

  auto start = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int bm = 0; bm < lb * lM; bm++) {
      int batch = bm / lM;
      int row   = bm % lM;
      int row_start = dOffsets[batch * (lM + 1) + row];
      int row_end   = dOffsets[batch * (lM + 1) + row + 1];
      for (int j = row_start; j < row_end; j++) {
        int col = dColumns[batch * lNnz + j];
        float val = 0.f;
        for (int l = 0; l < lK; l++)
          val += dA[batch * lM * lK + row * lK + l] *
                 dB[batch * lK * lN + l * lN + col];
        dValues[batch * lNnz + j] = val;
      }
    }
  }

  auto end  = std::chrono::steady_clock::now();
  double us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
              * 1e-3 / repeat;
  printf("Average execution time of SDDMM: %f (us)\n", us);

  if (verify) {
    #pragma omp target update from(dValues[0:bNnz])
    int correct = 1;
    for (int i = 0; i < b && correct; i++) {
      for (int j = 0; j < nnz; j++) {
        if (fabsf(dValues[i * nnz + j] - hC_result[i * nnz + j]) > 1e-2f) {
          printf("@batch%d index%d: %f != %f\n", i, j,
                 dValues[i * nnz + j], hC_result[i * nnz + j]);
          correct = 0;
          break;
        }
      }
    }
    if (correct)
      printf("sddmm_csr_batched_example test PASSED\n");
    else
      printf("sddmm_csr_batched_example test FAILED: wrong result\n");
  }

  #pragma omp target exit data map(delete: dA[0:bMA], dB[0:bMB], dOffsets[0:bOff], \
                                   dColumns[0:bNnz], dValues[0:bNnz])
  return 0;
}
