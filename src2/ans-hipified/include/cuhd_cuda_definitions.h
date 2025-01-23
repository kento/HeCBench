/*****************************************************************************
 *
 * MULTIANS - Massively parallel ANS decoding on GPUs
 *
 * released under LGPL-3.0
 *
 * 2017-2019 André Weißenberger
 *
 *****************************************************************************/

#ifndef CUHD_CUDA_DEFINITIONS_
#define CUHD_CUDA_DEFINITIONS_

#include <hip/hip_runtime.h>

#include <iostream>

// CUERR may be called for error checking
#define CUERR {                                                              \
    hipError_t err;                                                         \
    if ((err = hipGetLastError()) != hipSuccess) {                         \
       std::cout << "CUDA error: " << hipGetErrorString(err) << " : "       \
                 << __FILE__ << ", line " << __LINE__ << std::endl;          \
       exit(1);                                                              \
    }                                                                        \
}

#endif /* CUHD_CUDA_DEFINITIONS */
