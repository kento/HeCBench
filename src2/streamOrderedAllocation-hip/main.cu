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
 * This sample demonstrates stream ordered memory allocation on a GPU using
 * hipMallocAsync and cudaMemPool family of APIs.
 *
 * basicStreamOrderedAllocation(): demonstrates stream ordered allocation using
 * hipMallocAsync/hipFreeAsync APIs with default settings.
 *
 * streamOrderedAllocationPostSync(): demonstrates if there's a synchronization
 * in between allocations, then setting the release threshold on the pool will
 * make sure the synchronize will not free memory back to the OS.
 */

#include <stdio.h>
#include <climits>
#include <chrono>
#include <iostream>
#include <hip/hip_runtime.h>

#ifndef checkHipErrors
#define checkHipErrors(err)  __checkHipErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkHipErrors(hipError_t err, const char *file, const int line)
{   
  if (hipSuccess != err)
  {   
    std::cerr << "HIP Error = " << err << ": "
              << hipGetErrorString(err) << " from file "
              << file  << ", line " << line << std::endl;
  }
}
#endif


#define repeat 100

/* Add two vectors on the GPU */
__global__ void vectorAddGPU(const float *a, const float *b, float *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    c[idx] = a[idx] + b[idx];
  }
}

int basicAllocation(const int nelem, const float *a,
                    const float *b, float *c, bool timing) {
  float *d_a, *d_b, *d_c;  // Device buffers
  float errorNorm, refNorm, ref, diff;
  size_t bytes = nelem * sizeof(float);

  printf("Starting basicAllocation()\n");

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    checkHipErrors(hipMalloc(&d_a, bytes));
    checkHipErrors(hipMalloc(&d_b, bytes));
    checkHipErrors(hipMalloc(&d_c, bytes));
    checkHipErrors(
        hipMemcpy(d_a, a, bytes, hipMemcpyHostToDevice));
    checkHipErrors(
        hipMemcpy(d_b, b, bytes, hipMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((unsigned int)ceil(nelem / (float)block.x));
    vectorAddGPU<<<grid, block>>>(d_a, d_b, d_c, nelem);

    checkHipErrors(hipFree(d_a));
    checkHipErrors(hipFree(d_b));
    checkHipErrors(
        hipMemcpy(c, d_c, bytes, hipMemcpyDeviceToHost));
    checkHipErrors(hipFree(d_c));
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  if (timing)
    printf("Total elapsed time: %f (us) over %d iterations\n", (time * 1e-3f), repeat);
  else {
    /* Compare the results */
    printf("> Checking the results from vectorAddGPU() ...\n");
    errorNorm = 0.f;
    refNorm = 0.f;

    for (int n = 0; n < nelem; n++) {
      ref = a[n] + b[n];
      diff = c[n] - ref;
      errorNorm += diff * diff;
      refNorm += ref * ref;
    }

    errorNorm = (float)sqrt((double)errorNorm);
    refNorm = (float)sqrt((double)refNorm);
    if (errorNorm / refNorm < 1.e-6f)
      printf("basicAllocation PASSED\n");
  }

  return (timing || errorNorm / refNorm < 1.e-6f) ? EXIT_SUCCESS : EXIT_FAILURE;
}

int basicStreamOrderedAllocation(const int nelem, const float *a,
                                 const float *b, float *c, bool timing) {
  float *d_a, *d_b, *d_c;  // Device buffers
  float errorNorm, refNorm, ref, diff;
  size_t bytes = nelem * sizeof(float);

  hipStream_t stream;
  printf("Starting basicStreamOrderedAllocation()\n");
  checkHipErrors(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

  if (timing) {
    hipMemPool_t memPool;
    checkHipErrors(hipDeviceGetDefaultMemPool(&memPool, 0));
    uint64_t thresholdVal = 0;
    // set high release threshold on the default pool so that hipFreeAsync will
    // not actually release memory to the system. By default, the release
    // threshold for a memory pool is set to zero. This implies that the HIP
    // driver is allowed to release a memory chunk back to the system as long as
    // it does not contain any active suballocations.
    checkHipErrors(hipMemPoolSetAttribute(
        memPool, hipMemPoolAttrReleaseThreshold, (void *)&thresholdVal));
  }

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    checkHipErrors(hipMallocAsync(&d_a, bytes, stream));
    checkHipErrors(hipMallocAsync(&d_b, bytes, stream));
    checkHipErrors(hipMallocAsync(&d_c, bytes, stream));
    checkHipErrors(
        hipMemcpyAsync(d_a, a, bytes, hipMemcpyHostToDevice, stream));
    checkHipErrors(
        hipMemcpyAsync(d_b, b, bytes, hipMemcpyHostToDevice, stream));

    dim3 block(256);
    dim3 grid((unsigned int)ceil(nelem / (float)block.x));
    vectorAddGPU<<<grid, block, 0, stream>>>(d_a, d_b, d_c, nelem);

    checkHipErrors(hipFreeAsync(d_a, stream));
    checkHipErrors(hipFreeAsync(d_b, stream));
    checkHipErrors(
        hipMemcpyAsync(c, d_c, bytes, hipMemcpyDeviceToHost, stream));
    checkHipErrors(hipFreeAsync(d_c, stream));
    checkHipErrors(hipStreamSynchronize(stream));
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  if (timing)
    printf("Total elapsed time: %f (us) over %d iterations\n", (time * 1e-3f), repeat);
  else {
    /* Compare the results */
    printf("> Checking the results from vectorAddGPU() ...\n");
    errorNorm = 0.f;
    refNorm = 0.f;

    for (int n = 0; n < nelem; n++) {
      ref = a[n] + b[n];
      diff = c[n] - ref;
      errorNorm += diff * diff;
      refNorm += ref * ref;
    }

    errorNorm = (float)sqrt((double)errorNorm);
    refNorm = (float)sqrt((double)refNorm);
    if (errorNorm / refNorm < 1.e-6f)
      printf("basicStreamOrderedAllocation PASSED\n");
  }

  checkHipErrors(hipStreamDestroy(stream));

  return (timing || errorNorm / refNorm < 1.e-6f) ? EXIT_SUCCESS : EXIT_FAILURE;
}

// streamOrderedAllocationPostSync(): demonstrates If the application wants the
// memory to persist in the pool beyond synchronization, then it sets the
// release threshold on the pool. This way, when the application reaches the
// "steady state", it is no longer allocating/freeing memory from the OS.
int streamOrderedAllocationPostSync(const int nelem,
                                    const float *a, const float *b, float *c,
                                    bool timing) {
  float *d_a, *d_b, *d_c;  // Device buffers
  float errorNorm, refNorm, ref, diff;
  size_t bytes = nelem * sizeof(float);

  hipStream_t stream;
  printf("Starting streamOrderedAllocationPostSync()\n");
  checkHipErrors(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

  if (timing) {
    hipMemPool_t memPool;
    checkHipErrors(hipDeviceGetDefaultMemPool(&memPool, 0));
    uint64_t thresholdVal = ULONG_MAX;
    // set high release threshold on the default pool so that hipFreeAsync will
    // not actually release memory to the system. By default, the release
    // threshold for a memory pool is set to zero. This implies that the HIP
    // driver is allowed to release a memory chunk back to the system as long as
    // it does not contain any active suballocations.
    checkHipErrors(hipMemPoolSetAttribute(
        memPool, hipMemPoolAttrReleaseThreshold, (void *)&thresholdVal));
  }
  
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    checkHipErrors(hipMallocAsync(&d_a, bytes, stream));
    checkHipErrors(hipMallocAsync(&d_b, bytes, stream));
    checkHipErrors(hipMallocAsync(&d_c, bytes, stream));
    checkHipErrors(
        hipMemcpyAsync(d_a, a, bytes, hipMemcpyHostToDevice, stream));
    checkHipErrors(
        hipMemcpyAsync(d_b, b, bytes, hipMemcpyHostToDevice, stream));

    dim3 block(256);
    dim3 grid((unsigned int)ceil(nelem / (float)block.x));
    vectorAddGPU<<<grid, block, 0, stream>>>(d_a, d_b, d_c, nelem);

    checkHipErrors(hipFreeAsync(d_a, stream));
    checkHipErrors(hipFreeAsync(d_b, stream));
    checkHipErrors(
        hipMemcpyAsync(c, d_c, bytes, hipMemcpyDeviceToHost, stream));
    checkHipErrors(hipFreeAsync(d_c, stream));
    checkHipErrors(hipStreamSynchronize(stream));
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  if (timing)
    printf("Total elapsed time: %f (us) over %d iterations\n", (time * 1e-3f), repeat);
  else {
    /* Compare the results */
    printf("> Checking the results from vectorAddGPU() ...\n");
    errorNorm = 0.f;
    refNorm = 0.f;

    for (int n = 0; n < nelem; n++) {
      ref = a[n] + b[n];
      diff = c[n] - ref;
      errorNorm += diff * diff;
      refNorm += ref * ref;
    }

    errorNorm = (float)sqrt((double)errorNorm);
    refNorm = (float)sqrt((double)refNorm);
    if (errorNorm / refNorm < 1.e-6f)
      printf("streamOrderedAllocationPostSync PASSED\n");
  }

  checkHipErrors(hipStreamDestroy(stream));

  return (timing || errorNorm / refNorm < 1.e-6f) ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char **argv) {
  int nelem;
  size_t bytes;
  float *a, *b, *c;  // Host

/*
  int dev = 0;  // use default device 0
  int isMemPoolSupported = 0;
  checkHipErrors(hipDeviceGetAttribute(&isMemPoolSupported,
                                       hipDeviceAttributeMemoryPoolsSupported, dev));
  if (!isMemPoolSupported) {
    printf("Skip execution as device does not support Memory Pools\n");
    return 1;
  }
*/

  // Allocate CPU memory.
  nelem = 33554432; //1048576;
  bytes = nelem * sizeof(float);

  a = (float *)malloc(bytes);
  b = (float *)malloc(bytes);
  c = (float *)malloc(bytes);
  /* Initialize the vectors. */
  for (int n = 0; n < nelem; n++) {
    a[n] = rand() / (float)RAND_MAX;
    b[n] = rand() / (float)RAND_MAX;
  }

  // disable timing
  int ret0 = basicAllocation(nelem, a, b, c, false);
  int ret1 = basicStreamOrderedAllocation(nelem, a, b, c, false);
  int ret2 = streamOrderedAllocationPostSync(nelem, a, b, c, false);

  int r = ((ret0 == EXIT_SUCCESS &&
            ret1 == EXIT_SUCCESS && 
            ret2 == EXIT_SUCCESS) ? EXIT_SUCCESS : EXIT_FAILURE);

  // enable timing 
  basicAllocation(nelem, a, b, c, true);
  basicStreamOrderedAllocation(nelem, a, b, c, true);
  streamOrderedAllocationPostSync(nelem, a, b, c, true);

  /* Memory clean up */
  free(a);
  free(b);
  free(c);

  return r;
}
