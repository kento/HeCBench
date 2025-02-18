#include "hip/hip_runtime.h"
#include "hip/hip_runtime.h"
#include <sys/time.h>
#include <stdio.h>
#include <type_traits> // is_same
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if 0
#include <cuda_bf16.h>
#else
#include <hip/amd_detail/amd_hip_bf16.h>
#endif

#include <hipblaslt.h>
#include "utils.h"

// cuBLAS error checking
void cublasCheck(hipblasStatus_t status, const char *file, int line)
{
    if (status != HIPBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: status: %d, line: %d in %s\n", status, line, file);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4 is OK
static size_t cublaslt_workspace_size = 32 * 1024 * 1024;
static void* cublaslt_workspace = NULL;

hipblasLtHandle_t handle;

template <typename T, typename S>
void allocate_memory(int m, int n, int k, T **A, T **B, S **C) {
  hipMallocManaged(A, m * k * sizeof(T));
  hipMallocManaged(B, k * n * sizeof(T));
  hipMallocManaged(C, m * n * sizeof(S));
}

template <typename T, typename S>
void free_memory(T *A, T *B, S *C) {
  hipFree(A);
  hipFree(B);
  hipFree(C);
}

template <typename T, typename S, typename CT>
bool cublasLt_gemm(
    const int m, const int n, const int k,
    T *A, T *B, S *C,
    int lda, int ldb, int ldc,
    const CT *alpha, const CT *beta)
{
  bool status = true;
  hipblasLtMatmulDesc_t operationDesc;
  hipblasLtMatmulPreference_t preference;
  hipblasLtMatrixLayout_t ALayout;
  hipblasLtMatrixLayout_t BLayout;
  hipblasLtMatrixLayout_t CLayout;
  hipblasLtMatmulHeuristicResult_t heuristic;

  // create the operation descriptor
  //hipblasOperation_t opTranspose = HIPBLAS_OP_T;
  hipblasOperation_t opNoTranspose = HIPBLAS_OP_N;

  hipDataType AType, CType, SType;
  hipblasComputeType_t ComputeType;
  if (std::is_same<T, double>::value) {
    AType = SType = CType = HIP_R_64F;
    ComputeType = HIPBLAS_COMPUTE_64F;
  } else if (std::is_same<T, float>::value) {
    AType = SType = CType = HIP_R_32F;
    ComputeType = HIPBLAS_COMPUTE_32F;
  } else if (std::is_same<T, __half>::value) {
    AType = CType = HIP_R_16F;
    if (std::is_same<CT, __half>::value) {
      SType = HIP_R_16F;
      ComputeType = HIPBLAS_COMPUTE_16F;
    }
    else {
      SType = HIP_R_32F;
      ComputeType = HIPBLAS_COMPUTE_32F;
    }
  } else if (std::is_same<T, __hip_bfloat16>::value) {
    AType = CType = HIP_R_16BF;
    SType = HIP_R_32F;
    ComputeType = HIPBLAS_COMPUTE_32F;
  } else if (std::is_same<T, int8_t>::value) {
    AType = HIP_R_8I;
    CType = SType = HIP_R_32I;
    ComputeType = HIPBLAS_COMPUTE_32I;
  } else {
    printf("Not supported data type.");
    return false;
  }
  
  // compute type and scale type
  cublasCheck(hipblasLtMatmulDescCreate(&operationDesc, ComputeType, SType));
  cublasCheck(hipblasLtMatmulDescSetAttribute(operationDesc,
              HIPBLASLT_MATMUL_DESC_TRANSA, &opNoTranspose, sizeof(opNoTranspose)));
  cublasCheck(hipblasLtMatmulDescSetAttribute(
              operationDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opNoTranspose, sizeof(opNoTranspose)));

  // define matrix layouts
  cublasCheck(hipblasLtMatrixLayoutCreate(&ALayout, AType, m, k, lda));
  cublasCheck(hipblasLtMatrixLayoutCreate(&BLayout, AType, k, n, ldb));
  cublasCheck(hipblasLtMatrixLayoutCreate(&CLayout, CType, m, n, ldc));

  // create a preference handle with specified max workspace
  cublasCheck(hipblasLtMatmulPreferenceCreate(&preference));
  cublasCheck(hipblasLtMatmulPreferenceSetAttribute(preference,
      HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

  // find a suitable algorithm
  int returnedResults = 0;
  cublasCheck(hipblasLtMatmulAlgoGetHeuristic(handle, operationDesc,
      ALayout, BLayout, CLayout, CLayout,
      preference, 1, &heuristic, &returnedResults));
  if (returnedResults == 0) {
      printf("No cuBLASLt algorithm found.\n");
      status = false;
      goto cleanups;
  }

  // call the matmul
  cublasCheck(hipblasLtMatmul(handle, operationDesc,
      alpha, B, ALayout, A, BLayout, beta,
      C, CLayout, C, CLayout, &heuristic.algo,
      cublaslt_workspace, cublaslt_workspace_size, 0));

  // wait for matmul
  hipDeviceSynchronize();

  cleanups:
  cublasCheck(hipblasLtMatmulPreferenceDestroy(preference));
  cublasCheck(hipblasLtMatmulDescDestroy(operationDesc));
  cublasCheck(hipblasLtMatrixLayoutDestroy(ALayout));
  cublasCheck(hipblasLtMatrixLayoutDestroy(BLayout));
  cublasCheck(hipblasLtMatrixLayoutDestroy(CLayout));

  return status;
}

template <typename T, typename S, typename CT>
void test_gemm(//hipblasLtHandle_t handle,
  const int m,  const int n,  const int k,
  T *A, T *B, S *C,
  const CT *alpha, const CT *beta, const int iteration)
{
  double total_time = 0;
  struct timeval start, end;

  for (int i = 0; i < iteration; ++i) {
    hipDeviceSynchronize();
    gettimeofday(&start, NULL);
    bool success = cublasLt_gemm( n, // number of rows of matrix A and C
                                  m, // number of columns of matrix B and C
                                  k, // number of columns of A and rows of B
                                  B,
                                  A,
                                  C,
                                  n, // lda
                                  k, // ldb
                                  n, // ldc
                                  alpha,
                                  beta);
    hipDeviceSynchronize();
    gettimeofday(&end, NULL);

    if (!success) break;
    else if (i > 0) {
      total_time += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
    }
  }
  if (total_time > 0.0) {
    double avg_time = total_time / (iteration - 1);
    performance(m, n, k, std::is_same<T, int8_t>::value, avg_time * 1e-3);
  }
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <M> <N> <K> <iterations>\n", argv[0]);
    printf("C = A X B (A: M * K, B: K * N, C: M * N)\n");
    return 1;
  }
  const int m = atoi(argv[1]);
  const int n = atoi(argv[2]);
  const int k = atoi(argv[3]);
  const int iteration = atoi(argv[4]);

  printf("shape: (%d, %d) x (%d, %d)\n", m, k, k, n);

  const double d_alpha = 1.0, d_beta = 0.0;
  const float f_alpha = 1.f, f_beta = 0.f;
  const __half h_alpha = __float2half_rn(1.f),
               h_beta = __float2half_rn(0.f);
  const float h_alpha2 = 1.f, h_beta2 = 0.f;
  const float bf_alpha(1.f), bf_beta(0.f);
  const int32_t i_alpha = 1, i_beta = 0;

  double *dA, *dB, *dC;
  float *fA, *fB, *fC;
  __half *hA, *hB, *hC;
  __hip_bfloat16 *bfA, *bfB, *bfC;
  int8_t *iA, *iB; int32_t *iC;

  allocate_memory(m, n, k, &dA, &dB, &dC);
  allocate_memory(m, n, k, &fA, &fB, &fC);
  allocate_memory(m, n, k, &hA, &hB, &hC);
  allocate_memory(m, n, k, &bfA, &bfB, &bfC);
  allocate_memory(m, n, k, &iA, &iB, &iC);

  for (int i = 0; i < m * k; ++i) {
    dA[i] = double(i % 255 - 127) / 127;
    fA[i] = float(i % 255 - 127) / 127;
    hA[i] = __float2half_rn(fA[i]);
    bfA[i] = __float2bfloat16_rn(fA[i]);
    iA[i] = float2int8(fA[i], 127);
  }
  for (int i = 0; i < k * n; ++i) {
    dB[i] = double(i % 255 - 127) / 127;
    fB[i] = float(i % 255 - 127) / 127;
    hB[i] = __float2half_rn(fB[i]);
    bfB[i] = __float2bfloat16_rn(fB[i]);
    iB[i] = float2int8(fB[i], 127);
  }

  cublasCheck(hipblasLtCreate(&handle));
  hipMalloc(&cublaslt_workspace, cublaslt_workspace_size);

  printf(">>>>>>>>>>>>>>>>> test fp64 >>>>>>>>>>>>>>>>>\n");
  test_gemm(m, n, k, dA, dB, dC, &d_alpha, &d_beta, iteration);

  printf(">>>>>>>>>>>>>>>>> test fp32 >>>>>>>>>>>>>>>>>\n");
  test_gemm(m, n, k, fA, fB, fC, &f_alpha, &f_beta, iteration);

  printf(">>>>>>>>>>>>>>>>> test fp16 (compute type fp16) >>>>>>>>>>>>>>>>>\n");
  test_gemm(m, n, k, hA, hB, hC, &h_alpha, &h_beta, iteration);

  printf(">>>>>>>>>>>>>>>>> test fp16 (compute type fp32) >>>>>>>>>>>>>>>>>\n");
  test_gemm(m, n, k, hA, hB, hC, &h_alpha2, &h_beta2, iteration);

  printf(">>>>>>>>>>>>>>>>> test bfloat16 (compute type fp32) >>>>>>>>>>>>>>>>>\n");
  test_gemm(m, n, k, bfA, bfB, bfC, &bf_alpha, &bf_beta, iteration);

  printf(">>>>>>>>>>>>>>>>> test int8 >>>>>>>>>>>>>>>>>\n");
  test_gemm(m, n, k, iA, iB, iC, &i_alpha, &i_beta, iteration);

  printf(">>>>>>>>>>>>>>>>> compare first ten values >>>>>>>>>>>>>>>>>\n");
  printf("fp64: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5lf%c", dC[i], " \n"[i==9]);

  printf("fp32: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", fC[i], " \n"[i==9]);

  printf("fp16: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", float(hC[i]), " \n"[i==9]);

  printf("bf16: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", float(bfC[i]), " \n"[i==9]);

  printf("int8: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", float(iC[i])/127/127, " \n"[i==9]);

  hipFree(cublaslt_workspace);
  cublasCheck(hipblasLtDestroy(handle));
  free_memory(dA, dB, dC);
  free_memory(fA, fB, fC);
  free_memory(hA, hB, hC);
  free_memory(bfA, bfB, bfC);
  free_memory(iA, iB, iC);
  return 0;
}
