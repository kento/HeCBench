#ifndef stk_algsup_CudaCall_hpp
#define stk_algsup_CudaCall_hpp

#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>

//----------------------------------------------------------------
inline
void stk_cuda_call(hipError_t err , const char* name )
{
  if ( err != hipSuccess ) {
    fprintf(stderr, "%s error: %s\n",name, hipGetErrorString(err) );
    exit(-1);
  }
}

#define CUDA_CALL( cuda_fn ) stk_cuda_call( cuda_fn , #cuda_fn )


#endif

