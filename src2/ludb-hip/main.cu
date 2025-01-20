/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This example demonstrates how to use the hipBLAS library API
 * for lower-upper (LU) decomposition of a matrix. LU decomposition
 * factors a matrix as the product of upper triangular matrix and
 * lower trianglular matrix.
 *
 * https://en.wikipedia.org/wiki/LU_decomposition
 *
 * This sample uses 10000 matrices of size NxN and performs
 * LU decomposition of them using batched decomposition API
 * of hipBLAS library. To test the correctness of upper and lower
 * matrices generated, they are multiplied and compared with the
 * original input matrix.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <chrono>

// rocm libraries and helpers
#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>

#ifndef checkHipErrors
#define checkHipErrors(err)  __checkHipErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkHipErrors(hipError_t err, const char *file, const int line)
{
  if (hipSuccess != err)
  {
    printf("%s from file %s, line %d\n", hipGetErrorString(err), file, line);
  }
}
#endif

// configurable parameters
// dimension of matrix
#define N 48
#define BATCH_SIZE 10000

/* comment this to use single precision */
//#define DOUBLE_PRECISION 

#ifdef DOUBLE_PRECISION
#define DATA_TYPE double
#define MAX_ERROR 1e-15
#else
#define DATA_TYPE float
#define MAX_ERROR 1e-6
#endif /* DOUBLE_PRCISION */

// wrapper
hipblasStatus_t getrfBatched(hipblasHandle_t handle, int n,
                                   DATA_TYPE* const A[], int lda, int* P,
                                   int* info, int batchSize) {
#ifdef DOUBLE_PRECISION
  return hipblasDgetrfBatched(handle, n, A, lda, P, info, batchSize);
#else
  return hipblasSgetrfBatched(handle, n, A, lda, P, info, batchSize);
#endif
}

// wrapper around malloc
// clears the allocated memory to 0
// terminates the program if malloc fails
void* xmalloc(size_t size) {
  void* ptr = malloc(size);
  if (ptr == NULL) {
    printf("> ERROR: malloc for size %zu failed..\n", size);
    exit(EXIT_FAILURE);
  }
  memset(ptr, 0, size);
  return ptr;
}

// initalize identity matrix
void initIdentityMatrix(DATA_TYPE* mat) {
  // clear the matrix
  memset(mat, 0, N * N * sizeof(DATA_TYPE));

  // set all diagonals to 1
  for (int i = 0; i < N; i++) {
    mat[(i * N) + i] = 1.0;
  }
}

// initialize matrix with all elements as 0
void initZeroMatrix(DATA_TYPE* mat) {
  memset(mat, 0, N * N * sizeof(DATA_TYPE));
}

// fill random value in column-major matrix
void initRandomMatrix(DATA_TYPE* mat) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      mat[(j * N) + i] =
          (DATA_TYPE)1.0 + ((DATA_TYPE)rand() / (DATA_TYPE)RAND_MAX);
    }
  }

  // diagonal dominant matrix to insure it is invertible matrix
  for (int i = 0; i < N; i++) {
    mat[(i * N) + i] += (DATA_TYPE)N;
  }
}

// print column-major matrix
void printMatrix(DATA_TYPE* mat) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%20.16f ", mat[(j * N) + i]);
    }
    printf("\n");
  }
  printf("\n");
}

// matrix mulitplication
void matrixMultiply(DATA_TYPE* res, DATA_TYPE* mat1, DATA_TYPE* mat2) {
  initZeroMatrix(res);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        res[(j * N) + i] += mat1[(k * N) + i] * mat2[(j * N) + k];
      }
    }
  }
}

// check matrix equality
bool checkRelativeError(DATA_TYPE* mat1, DATA_TYPE* mat2, DATA_TYPE maxError) {
  DATA_TYPE err = (DATA_TYPE)0.0;
  DATA_TYPE refNorm = (DATA_TYPE)0.0;
  DATA_TYPE relError = (DATA_TYPE)0.0;
  DATA_TYPE relMaxError = (DATA_TYPE)0.0;

  for (int i = 0; i < N * N; i++) {
    refNorm = abs(mat1[i]);
    err = abs(mat1[i] - mat2[i]);

    if (refNorm != 0.0 && err > 0.0) {
      relError = err / refNorm;
      relMaxError = relMaxError > relError ? relMaxError : relError;
    }

    if (relMaxError > maxError) return false;
  }
  return true;
}

// decode lower and upper matrix from single matrix
// returned by getrfBatched()
void getLUdecoded(DATA_TYPE* mat, DATA_TYPE* L, DATA_TYPE* U) {
  // init L as identity matrix
  initIdentityMatrix(L);

  // copy lower triangular values from mat to L (skip diagonal)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < i; j++) {
      L[(j * N) + i] = mat[(j * N) + i];
    }
  }

  // init U as all zero
  initZeroMatrix(U);

  // copy upper triangular values from mat to U
  for (int i = 0; i < N; i++) {
    for (int j = i; j < N; j++) {
      U[(j * N) + i] = mat[(j * N) + i];
    }
  }
}

// generate permutation matrix from pivot vector
void getPmatFromPivot(DATA_TYPE* Pmat, int* P) {
  int pivot[N];

  // pivot vector in base-1
  // convert it to base-0
  for (int i = 0; i < N; i++) {
    P[i]--;
  }

  // generate permutation vector from pivot
  // initialize pivot with identity sequence
  for (int k = 0; k < N; k++) {
    pivot[k] = k;
  }

  // swap the indices according to pivot vector
  for (int k = 0; k < N; k++) {
    int q = P[k];

    // swap pivot(k) and pivot(q)
    int s = pivot[k];
    int t = pivot[q];
    pivot[k] = t;
    pivot[q] = s;
  }

  // generate permutation matrix from pivot vector
  initZeroMatrix(Pmat);
  for (int i = 0; i < N; i++) {
    int j = pivot[i];
    Pmat[(j * N) + i] = (DATA_TYPE)1.0;
  }
}

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  // hipBLAS variables
  hipblasStatus_t status;
  hipblasHandle_t handle;

  // host variables
  size_t matSize = N * N * sizeof(DATA_TYPE);

  DATA_TYPE* h_AarrayInput;
  DATA_TYPE* h_AarrayOutput;
  DATA_TYPE* h_ptr_array[BATCH_SIZE];

  int* h_pivotArray;
  int* h_infoArray;

  // device variables
  DATA_TYPE* d_Aarray;
  DATA_TYPE** d_ptr_array;

  int* d_pivotArray;
  int* d_infoArray;

  int err_count = 0;

  // seed the rand() function with time
  srand(12345);

  // initialize hipBLAS
  printf("> initializing..\n");
  status = hipblasCreate(&handle);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    printf("> ERROR: hipBLAS initialization failed..\n");
    return (EXIT_FAILURE);
  }

#ifdef DOUBLE_PRECISION
  printf("> using DOUBLE precision..\n");
#else
  printf("> using SINGLE precision..\n");
#endif

  printf("> pivot ENABLED..\n");

  // allocate memory for host variables
  h_AarrayInput = (DATA_TYPE*)xmalloc(BATCH_SIZE * matSize);
  h_AarrayOutput = (DATA_TYPE*)xmalloc(BATCH_SIZE * matSize);

  h_pivotArray = (int*)xmalloc(N * BATCH_SIZE * sizeof(int));
  h_infoArray = (int*)xmalloc(BATCH_SIZE * sizeof(int));

  // allocate memory for device variables
  checkHipErrors(hipMalloc((void**)&d_Aarray, BATCH_SIZE * matSize));
  checkHipErrors(
      hipMalloc((void**)&d_pivotArray, N * BATCH_SIZE * sizeof(int)));
  checkHipErrors(hipMalloc((void**)&d_infoArray, BATCH_SIZE * sizeof(int)));
  checkHipErrors(
      hipMalloc((void**)&d_ptr_array, BATCH_SIZE * sizeof(DATA_TYPE*)));

  // fill matrix with random data
  printf("> generating random matrices..\n");
  for (int i = 0; i < BATCH_SIZE; i++) {
    initRandomMatrix(h_AarrayInput + (i * N * N));
  }

  // create pointer array for matrices
  for (int i = 0; i < BATCH_SIZE; i++) h_ptr_array[i] = d_Aarray + (i * N * N);

  // copy pointer array to device memory
  checkHipErrors(hipMemcpy(d_ptr_array, h_ptr_array,
                             BATCH_SIZE * sizeof(DATA_TYPE*),
                             hipMemcpyHostToDevice));

  long time = 0;
  // perform LU decomposition
  printf("> performing batched LU decomposition..\n");
  for (int i = 0; i <= repeat; i++) {
    // copy data to device from host
    //printf("> copying data from host memory to GPU memory..\n");
    checkHipErrors(hipMemcpy(d_Aarray, h_AarrayInput, BATCH_SIZE * matSize,
                               hipMemcpyHostToDevice));

    hipDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();
    status = getrfBatched(handle, N, d_ptr_array, N, d_pivotArray,
                          d_infoArray, BATCH_SIZE);
    hipDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    if (i != 0)
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  if (status != HIPBLAS_STATUS_SUCCESS) {
    printf("> ERROR: hipblasDgetrfBatched() failed with error %s..\n",
           hipblasStatusToString(status));
  } else {
    printf("Average kernel execution time : %f (us)\n", (time * 1e-3f) / repeat);
  }

  // copy data to host from device
  //printf("> copying data from GPU memory to host memory..\n");
  checkHipErrors(hipMemcpy(h_AarrayOutput, d_Aarray, BATCH_SIZE * matSize,
                           hipMemcpyDeviceToHost));
  checkHipErrors(hipMemcpy(h_infoArray, d_infoArray, BATCH_SIZE * sizeof(int),
                           hipMemcpyDeviceToHost));
  checkHipErrors(hipMemcpy(h_pivotArray, d_pivotArray,
                           N * BATCH_SIZE * sizeof(int),
                           hipMemcpyDeviceToHost));

  // verify the result
  printf("> verifying the result..\n");
  for (int i = 0; i < BATCH_SIZE; i++) {
    if (h_infoArray[i] == 0) {
      DATA_TYPE* A = h_AarrayInput + (i * N * N);
      DATA_TYPE* LU = h_AarrayOutput + (i * N * N);
      DATA_TYPE L[N * N];
      DATA_TYPE U[N * N];
      getLUdecoded(LU, L, U);

      // test P * A = L * U
      int* P = h_pivotArray + (i * N);
      DATA_TYPE Pmat[N * N];
      getPmatFromPivot(Pmat, P);

      // perform matrix multiplication
      DATA_TYPE PxA[N * N];
      DATA_TYPE LxU[N * N];
      matrixMultiply(PxA, Pmat, A);
      matrixMultiply(LxU, L, U);

      // check for equality of matrices
      if (!checkRelativeError(PxA, LxU, (DATA_TYPE)MAX_ERROR)) {
        printf("> ERROR: accuracy check failed for matrix number %05d..\n",
               i + 1);
        err_count++;
      }

    } else if (h_infoArray[i] > 0) {
      printf(
          "> execution for matrix %05d is successful, but U is singular and "
          "U(%d,%d) = 0..\n",
          i + 1, h_infoArray[i] - 1, h_infoArray[i] - 1);
    } else  // (h_infoArray[i] < 0)
    {
      printf("> ERROR: matrix %05d have an illegal value at index %d = %lf..\n",
             i + 1, -h_infoArray[i],
             *(h_AarrayInput + (i * N * N) + (-h_infoArray[i])));
    }
  }

  // free device variables
  checkHipErrors(hipFree(d_ptr_array));
  checkHipErrors(hipFree(d_infoArray));
  checkHipErrors(hipFree(d_pivotArray));
  checkHipErrors(hipFree(d_Aarray));

  // free host variables
  if (h_infoArray) free(h_infoArray);
  if (h_pivotArray) free(h_pivotArray);
  if (h_AarrayOutput) free(h_AarrayOutput);
  if (h_AarrayInput) free(h_AarrayInput);

  // destroy hipBLAS handle
  status = hipblasDestroy(handle);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    printf("> ERROR: hipBLAS uninitialization failed..\n");
    return (EXIT_FAILURE);
  }

  if (err_count > 0) {
    printf("> TEST FAILED for %d matrices, with precision: %g\n", err_count,
           MAX_ERROR);
    return (EXIT_FAILURE);
  }

  printf("> TEST SUCCESSFUL, with precision: %g\n", MAX_ERROR);
  return (EXIT_SUCCESS);
}
