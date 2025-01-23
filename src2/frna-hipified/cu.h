#include "hip/hip_runtime.h"
#ifndef CU_H
#define CU_H

#ifdef __HIPCC__

#define CU(x) if ((x) != hipSuccess) {;die("%s:%d: CUDA error: %s", __FILE__, __LINE__,hipGetErrorString(x)); }
#define DEV __device__
#define HOST __host__
#define GLOBAL __global__

#else

#define DEV
#define HOST
#define GLOBAL

#endif /* __HIPCC__ */

#endif /* CU_H */
