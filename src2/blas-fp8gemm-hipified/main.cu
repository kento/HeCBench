#include <chrono>
#include "helper.h"

/// Sample wrapper executing fp8 matmul with hipblasLtMatmul, with addition of per-tensor scaling, amax calculations, and
/// the workspace to support split-K algorithms.
///
/// pointer mode is for alpha and beta is always host, to change it configure the appropriate matmul descriptor
/// attribute matmul is not using cublas handle's configuration of math mode, here tensor ops are implicitly allowed; to
/// change this configure appropriate attribute in the preference handle
void LtFp8Matmul(const int repeat,
                 hipblasLtHandle_t ltHandle,
                 int m,
                 int n,
                 int k,
                 const float *alpha, /* host pointer */
                 const float *beta, /* host pointer */
                 const float *a_scale, /* device pointer */
                 const __nv_fp8_e4m3 *A,
                 int lda,
                 const float *b_scale, /* device pointer */
                 const __nv_fp8_e4m3 *B,
                 int ldb,
                 const float *c_scale, /* device pointer */
                 const hip_bfloat16 *C,
                 int ldc,
                 const float *d_scale, /* device pointer */
                 __nv_fp8_e4m3 *D,
                 float *amax_d, /* device pointer */
                 void *workspace,
                 size_t workspaceSize) {
    hipblasLtMatmulDesc_t operationDesc = NULL;
    hipblasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
    hipblasLtMatmulPreference_t preference = NULL;

    hipblasOperation_t transa = HIPBLAS_OP_T;
    hipblasOperation_t transb = HIPBLAS_OP_N;

    int returnedResults                             = 0;
    hipblasLtMatmulHeuristicResult_t heuristicResult = {};

    // create operation desciriptor; see hipblasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    checkCublasStatus(hipblasLtMatmulDescCreate(&operationDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    checkCublasStatus(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // set scaling factors
    checkCublasStatus(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
    checkCublasStatus(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));
    checkCublasStatus(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER, &c_scale, sizeof(c_scale)));
    checkCublasStatus(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_scale, sizeof(d_scale)));
    checkCublasStatus(hipblasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &amax_d, sizeof(amax_d)));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    // table of supported type combinations can be found in the documentation: https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul
    checkCublasStatus(hipblasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, transa == HIPBLAS_OP_N ? m : k, transa == HIPBLAS_OP_N ? k : m, lda));
    checkCublasStatus(hipblasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, transb == HIPBLAS_OP_N ? k : n, transb == HIPBLAS_OP_N ? n : k, ldb));
    checkCublasStatus(hipblasLtMatrixLayoutCreate(&Cdesc, HIP_R_16BF, m, n, ldc));
    checkCublasStatus(hipblasLtMatrixLayoutCreate(&Ddesc, CUDA_R_8F_E4M3, m, n, ldc));

    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from hipMalloc)
    checkCublasStatus(hipblasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(hipblasLtMatmulPreferenceSetAttribute(preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
    // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
    checkCublasStatus(hipblasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        printf("no heuristic function available for current configuration\n");
        return;
    }

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      checkCublasStatus(hipblasLtMatmul(ltHandle,
                                     operationDesc,
                                     alpha, A, Adesc,
                                     B, Bdesc, beta,
                                     C, Cdesc,
                                     D, Ddesc,
                                     &heuristicResult.algo,
                                     workspace,
                                     workspaceSize,
                                     0));
    }

    hipDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average hipblasLtMatmul execution time %f (us)\n", (time * 1e-3f) / repeat);

    // descriptors are no longer needed as all GPU work was already enqueued
    if (preference) checkCublasStatus(hipblasLtMatmulPreferenceDestroy(preference));
    if (Ddesc) checkCublasStatus(hipblasLtMatrixLayoutDestroy(Ddesc));
    if (Cdesc) checkCublasStatus(hipblasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(hipblasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(hipblasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) checkCublasStatus(hipblasLtMatmulDescDestroy(operationDesc));
}


int main(int argc, char *argv[])
{
   if (argc != 2) {
     printf("Usage: %s <repeat>\n", argv[0]);
     return 1;
   }
   const int repeat = atoi(argv[1]);


   TestBench<__nv_fp8_e4m3, 
             hip_bfloat16, // hipblasLtMatrixLayoutCreate
             __nv_fp8_e4m3,
             float> props(64, 128, 256, 2.0f, 1.0f, 32ULL * 1024 * 1024);

   props.run([&props, repeat] {
        LtFp8Matmul(repeat,
                    props.ltHandle,
                    props.m,
                    props.n,
                    props.k,
                    &props.alpha,
                    &props.beta,
                    props.AscaleDev, props.Adev, props.k,
                    props.BscaleDev, props.Bdev, props.k,
                    props.CscaleDev, props.Cdev, props.m,
                    props.DscaleDev, props.Ddev,
                    props.DamaxDev,
                    props.workspace,
                    props.workspaceSize);
    });

    return 0;
}
