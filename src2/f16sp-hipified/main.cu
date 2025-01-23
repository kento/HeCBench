#include "hip/hip_runtime.h"
#include "hip/hip_runtime.h"
/**
 * Copyright 2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipcub/hipcub.hpp>
#include <hipblas.h>
#include "kernels.h"

// The analytical result of dot product is 65504
void generateInput(half2 * a, size_t size)
{
  for (size_t i = 0; i < size; ++i)
  {
    half2 temp;
    temp.x = (half)(sqrt(32752.0/size));
    temp.y = (half)(sqrt(32752.0/size));
    a[i] = temp;
  }
}

int main(int argc, char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const size_t size = NUM_OF_BLOCKS*NUM_OF_THREADS;
  const size_t size_bytes = size * sizeof(half2);
  const size_t result_bytes = sizeof(float);
  const size_t result2_bytes = sizeof(half);

  half2 *a, *b;
  half2 *d_a, *d_b;

  float r, *d_r;
  half r2, *d_r2;

  a = (half2*) malloc (size_bytes);
  b = (half2*) malloc (size_bytes);
  hipMalloc((void**)&d_a, size_bytes);
  hipMalloc((void**)&d_b, size_bytes);

  hipMalloc((void**)&d_r, result_bytes);
  hipMalloc((void**)&d_r2, result2_bytes);

  srand(123);
  generateInput(a, size);
  hipMemcpy(d_a, a, size_bytes, hipMemcpyHostToDevice);

  generateInput(b, size);
  hipMemcpy(d_b, b, size_bytes, hipMemcpyHostToDevice);

  printf("\nNumber of elements in the vectors is %zu\n", size * 2);

  // evaluate the impact of grid sizes on performance
  for (int grid = NUM_OF_BLOCKS; grid >= NUM_OF_BLOCKS / 16; grid = grid / 2) {

    printf("\nGPU grid size is %d\n", grid);

    // warmup
    for (int i = 0; i < 1000; i++)
      scalarProductKernel_intrinsics<<<grid, NUM_OF_THREADS>>>(d_a, d_b, d_r, size);

    hipDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      hipMemset(d_r, 0, result_bytes);
      scalarProductKernel_intrinsics<<<grid, NUM_OF_THREADS>>>(d_a, d_b, d_r, size);
    }

    hipDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel1 execution time %f (us)\n", (time * 1e-3f) / repeat);

    hipMemcpy(&r, d_r, result_bytes, hipMemcpyDeviceToHost);
    printf("Error rate: %e\n", fabsf(r - 65504.f)/65504.f);

    // warmup
    for (int i = 0; i < 1000; i++)
      scalarProductKernel_native_fp32<<<grid, NUM_OF_THREADS>>>(d_a, d_b, d_r, size);

    hipDeviceSynchronize();
    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      hipMemset(d_r, 0, result_bytes);
      scalarProductKernel_native_fp32<<<grid, NUM_OF_THREADS>>>(d_a, d_b, d_r, size);
    }

    hipDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel2 execution time %f (us)\n", (time * 1e-3f) / repeat);

    hipMemcpy(&r, d_r, result_bytes, hipMemcpyDeviceToHost);
    printf("Error rate: %e\n", fabsf(r - 65504.f)/65504.f);

    // warmup
    for (int i = 0; i < 1000; i++)
      scalarProductKernel_native2_fp32<<<grid, NUM_OF_THREADS>>>(d_a, d_b, d_r, size);

    hipDeviceSynchronize();
    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      hipMemset(d_r, 0, result_bytes);
      scalarProductKernel_native2_fp32<<<grid, NUM_OF_THREADS>>>(d_a, d_b, d_r, size);
    }

    hipDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel3 execution time %f (us)\n", (time * 1e-3f) / repeat);

    hipMemcpy(&r, d_r, result_bytes, hipMemcpyDeviceToHost);
    printf("Error rate: %e\n", fabsf(r - 65504.f)/65504.f);
  }

  printf("\n");
  // library-based dot product
  hipblasHandle_t h;
  hipblasCreate(&h);
  hipblasSetPointerMode(h, HIPBLAS_POINTER_MODE_DEVICE);

  hipDataType xType, yType, rType, eType;
  xType = yType = rType = HIP_R_16F;
  eType = HIP_R_32F;

  // warmup
  for (int i = 0; i < 1000; i++) {
    hipblasDotEx_v2(h, size*2, (half*)d_a, xType, 1, (half*)d_b,
                yType, 1, d_r2, rType, eType);
  }
  hipDeviceSynchronize();

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    hipblasDotEx_v2(h, size*2, (half*)d_a, xType, 1, (half*)d_b,
                yType, 1, d_r2, rType, eType);
  }
  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel (cublasDot) execution time %f (us)\n", (time * 1e-3f) / repeat);
  hipMemcpy(&r2, d_r2, result2_bytes, hipMemcpyDeviceToHost);
  printf("Error rate: %e\n", fabsf((float)r2 - 65504.f)/65504.f);

  hipFree(d_a);
  hipFree(d_b);
  hipFree(d_r);
  hipFree(d_r2);
  hipblasDestroy(h);
  free(a);
  free(b);

  return EXIT_SUCCESS;
}
