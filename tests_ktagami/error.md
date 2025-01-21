# halo-finder-cuda
## make
```
nvcc -Idfft -DPENCIL=1 -O3 -DRCB_UNTHREADED_BUILD -DUSE_SERIAL_COSMO  -I/usr/include/openmpi-x86_64/ -DID_64 -DPOSVEL_32 -DGRID_32 -DLONG_INTEGER  -I. -x cu -Idfft -c -o cuda/ForceTreeTest.o ForceTreeTest.cxx
In file included from ForceTreeTest.cxx:23:
./Partition.h:68:10: fatal error: mpi.h: No such file or directory
   68 | #include <mpi.h>
      |          ^~~~~~~
compilation terminated.
make: *** [Makefile:100: cuda/ForceTreeTest.o] Error 1
```

# haversine-omp
##  make -f Makefile.aomp
```
clang++  -std=c++14 -Wall -I../haversine-cuda/ -O3 -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 -c distance.cpp -o distance.o
In file included from distance.cpp:1:
In file included from ../haversine-cuda/distance.h:5:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/math.h:20:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/cmath:86:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/__clang_hip_cmath.h:20:
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/type_traits:2360:21: error: static assertion failed due to requirement '__declval_protector<long>::__stop': declval() must not be used!
 2360 |       static_assert(__declval_protector<_Tp>::__stop,
      |
(中略)
7 errors generated.
make: *** [Makefile.aomp:64: distance.o] Error 1
```

# heartwall-omp
## 実行時エラー (nvc)
```
Workgroup size of kernel = 256
frame progress: Failing in Thread:1
Accelerator Fatal Error: call to cuMemcpyDtoHAsync returned error 719 (CUDA_ERROR_LAUNCH_FAILED): Launch failed (often invalid pointer dereference)
 File: /hs/work0/home/users/u0001620/work3/heartwall/heartwall-omp-nvc/./kernel/kernel.cpp
 Function: _Z18kernel_gpu_wrapper13params_commonPiS0_S0_S0_S0_S0_S0_S0_P5avi_t:20
 Line: 492
```

# heat2d-cuda
## make
```
nvcc  -std=c++14 -Xcompiler -Wall -Xcompiler -fopenmp -arch=sm_60 -O3 -c main.cu -o main.o
In file included from main.cu:16:
lapl_ss.c:1:10: fatal error: xmmintrin.h: No such file or directory
    1 | #include <xmmintrin.h>
      |          ^~~~~~~~~~~~~
compilation terminated.
make: *** [Makefile:51: main.o] Error 1
```
# heat2d-hip
## make
```
main.cu:106:3: error: expected ')'
  106 |          Lx*Ly*sizeof(float)*2.0/(t0*1.0e3),
      |
(中略)
23 warnings and 1 error generated when compiling for gfx90a.
failed to execute:/opt/rocm-6.2.0/lib/llvm/bin/clang++  --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a  -std=c++14 -Wall -fopenmp -O3 -c -x hip main.cu -o "main.o"
make: *** [Makefile:50: main.o] Error 1
```

# henry-omp
## make -f Makefile.aomp

```
clang++  -std=c++14 -Wall  -O3 -ffast-math -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 -c main.cpp -o main.o
In file included from main.cpp:3:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/math.h:20:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/cmath:86:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/__clang_hip_cmath.h:20:
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/type_traits:2360:21: error: static assertion failed due to requirement '__declval_protector<std::_Rb_tree_node_base *const &>::__stop': declval() must not be used!
 2360 |       static_assert(__declval_protector<_Tp>::__stop,
      |
(中略)
18 errors generated.
make: *** [Makefile.aomp:61: main.o] Error 1
```

# hexciton-hipified
## make
```
hipcc  -std=c++14  -Wall  -O3 -c main.cu -o main.o
In file included from main.cu:6:
In file included from ./utils.hpp:44:
./helper_math.h:1455:30: error: use of overloaded operator '/' is ambiguous (with operand types 'float2' (aka 'HIP_vector_type<float, 2>') and 'float2')
 1455 |     float2 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
      |
(中略)
fatal error: too many errors emitted, stopping now [-ferror-limit=]
50 warnings and 20 errors generated when compiling for gfx90a.
failed to execute:/opt/rocm-6.2.0/lib/llvm/bin/clang++  --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a  -std=c++14 -Wall -O3 -c -x hip main.cu -o "main.o"
make: *** [Makefile:51: main.o] Error 1
```

# histogram-omp
## make -f Makefile.aomp
```
clang++  -std=c++14 -Wall  -O3 -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march= -c histogram_compare_base.cpp -o histogram_compare_base.o
clang++: error: invalid target ID ''; format is a processor name followed by an optional colon-delimited list of features followed by an enable/disable sign (e.g., 'gfx908:sramecc+:xnack-')
make: *** [Makefile.aomp:60: histogram_compare_base.o] Error 1
```

# hwt1d-omp
## make -f Makefile.aomp

```
In file included from main.cpp:17:
In file included from ./hwt.h:28:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/math.h:20:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/cmath:86:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/__clang_hip_cmath.h:20:
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/type_traits:2360:21: error: static assertion failed due to requirement '__declval_protector<long>::__stop': declval() must not be used!
 2360 |       static_assert(__declval_protector<_Tp>::__stop,
      |
(中略)
8 errors generated.
make: *** [Makefile.aomp:61: main.o] Error 1
```


# hybridsort-omp
## make -f Makefile.aomp

```
clang++  -std=c++11 -Wall -O3 -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march= -o hybridsort.o -c hybridsort.c
clang++: warning: treating 'c' input as 'c++' when in C++ mode, this behavior is deprecated [-Wdeprecated]
clang++: error: invalid target ID ''; format is a processor name followed by an optional colon-delimited list of features followed by an enable/disable sign (e.g., 'gfx908:sramecc+:xnack-')
make: *** [Makefile.aomp:55: hybridsort.o] Error 1
```

# intrinsics-simd-hipified
## make
```
hipcc  -std=c++14  -Wall  -O3 -c main.cu -o main.o
main.cu:21:8: error: use of undeclared identifier '__vabs2'
   21 |   r  = __vabs2(a);
      |
(中略)
fatal error: too many errors emitted, stopping now [-ferror-limit=]
1 warning and 20 errors generated when compiling for gfx90a.
failed to execute:/opt/rocm-6.2.0/lib/llvm/bin/clang++  --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a  -std=c++14 -Wall -O3 -c -x hip main.cu -o "main.o"
make: *** [Makefile:50: main.o] Error 1
```
# jaccard-hipified
## make
```
hipcc  -std=c++14  -Wall  -O3 -c main.cu -o main.o
main.cu:59:12: error: use of undeclared identifier '__shfl_sync'
   59 |     last = __shfl_sync(mask, sum, blockDim.x-1, blockDim.x);
      |
(中略)
4 errors generated when compiling for gfx90a.
failed to execute:/opt/rocm-6.2.0/lib/llvm/bin/clang++  --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a  -std=c++14 -Wall -O3 -c -x hip main.cu -o "main.o"
make: *** [Makefile:51: main.o] Error 1
```

# jacobi-hipified
## make
```
ipcc  -std=c++14  -Wall  -O3 -c main.cu -o main.o
main.cu:102:12: error: use of undeclared identifier '__shfl_down_sync'; did you mean '__shfl_down'?
  102 |     err += __shfl_down_sync(0xffffffff, err, offset);
      |            ^~~~~~~~~~~~~~~~
      |            __shfl_down
/opt/rocm-6.2.0/include/hip/amd_detail/amd_warp_functions.h:342:14: note: '__shfl_down' declared here
  342 | unsigned int __shfl_down(unsigned int var, unsigned int lane_delta, int width = warpSize) {
      |              ^
(中略)
2 errors generated when compiling for gfx90a.
failed to execute:/opt/rocm-6.2.0/lib/llvm/bin/clang++  --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a  -std=c++14 -Wall -O3 -c -x hip main.cu -o "main.o"
make: *** [Makefile:52: main.o] Error 1
```

# kalman-omp
## 実行時エラー ( nvc )
```
./main 1000 100 3 100
Fatal error: expression 'HX_CU_CALL_CHECK(p_cuStreamSynchronize(stream))' (value 1) is not equal to expression 'HX_SUCCESS' (value 0)
Aborted (core dumped)
```

# kalman-omp
## make -f Makefile.aomp
```
In file included from main.cpp:19:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/math.h:20:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/cmath:86:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/__clang_hip_cmath.h:20:
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/type_traits:2360:21: error: static assertion failed due to requirement '__declval_protector<long>::__stop': declval() must not be used!
 2360 |       static_assert(__declval_protector<_Tp>::__stop,
      |
(中略)
7 errors generated.
make: *** [Makefile.aomp:60: main.o] Error 1
```

# keogh-omp
## make -f Makefile.aomp
```
In file included from main.cpp:3:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/math.h:20:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/cmath:86:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/__clang_hip_cmath.h:20:
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/type_traits:2360:21: error: static assertion failed due to requirement '__declval_protector<long>::__stop': declval() must not be used!
 2360 |       static_assert(__declval_protector<_Tp>::__stop,
      |                     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

7 errors generated.
make: *** [Makefile.aomp:60: main.o] Error 1
```

# kernelLaunch-cuda
## make

```
nvcc  -std=c++14 -Xcompiler -Wall -arch=sm_60 -O3 -c main.cu -o main.o
main.cu(49): Error: Formal parameter space overflowed (4104 bytes required, max 4096 bytes allowed) in function _Z19KernelWithLargeArgs15LargeKernelArgsPc

make: *** [Makefile:50: main.o] Error 1
```

# kmc-hip,  kmc-hipified
## make

```
hipcc  -std=c++14 -Wall -Wno-unused-result -I../kmc-cuda -O3 -c main.cpp -o main.o
main.cpp:3:10: fatal error: 'hipblas.h' file not found
    3 | #include <hipblas.h>
      |          ^~~~~~~~~~~
1 error generated when compiling for gfx90a.
failed to execute:/opt/rocm-6.2.0/lib/llvm/bin/clang++  --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a  -std=c++14 -Wall -Wno-unused-result -I../kmc-cuda -O3 -c -x hip main.cpp -o "main.o"
make: *** [Makefile:47: main.o] Error 1
```

# kmeans-omp
## make -f Makefile.aomp
```
In file included from read_input.c:38:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/math.h:20:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/cmath:86:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/__clang_hip_cmath.h:20:
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/type_traits:2360:21: error: static assertion failed due to requirement '__declval_protector<long>::__stop': declval() must not be used!
 2360 |       static_assert(__declval_protector<_Tp>::__stop,
      |
(中略)
1 warning and 8 errors generated.
make: *** [Makefile.aomp:46: read_input.o] Error 1
```

# langevin-omp
## make -f Makefile.aomp
```
In file included from main.cpp:3:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/math.h:20:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/cmath:86:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/__clang_hip_cmath.h:20:
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/type_traits:2360:21: error: static assertion failed due to requirement '__declval_protector<long>::__stop': declval() must not be used!
 2360 |       static_assert(__declval_protector<_Tp>::__stop,
      |
(中略)
7 errors generated.
make: *** [Makefile.aomp:60: main.o] Error 1
```

# langford-omp
## make -f Makefile.nvc
```
NVC++-W-0155-Compiler failed to translate accelerator region (see -Minfo messages): Missing branch target block (main.cpp: 356)
(中略)
NVC++-F-0704-Compilation aborted due to previous errors. (main.cpp)
NVC++/arm Linux 24.9-0: compilation aborted
make: *** [Makefile.nvc:57: main.o] Error 2
```

# laplace3d-omp
## make -f Makefile.aomp
```
In file included from main.cpp:8:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/math.h:20:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/cmath:86:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/__clang_hip_cmath.h:20:
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/type_traits:2360:21: error: static assertion failed due to requirement '__declval_protector<long>::__stop': declval() must not be used!
 2360 |       static_assert(__declval_protector<_Tp>::__stop,
      |
(中略)
8 errors generated.
make: *** [Makefile.aomp:61: main.o] Error 1
```

# layernorm-hipified
## make
```
hipcc  -std=c++14  -Wall  -O3 -c main.cu -o main.o
In file included from main.cu:10:
./common.h:5:10: fatal error: 'cooperative_groups/reduce.h' file not found
    5 | #include <cooperative_groups/reduce.h>
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1 error generated when compiling for gfx90a.
failed to execute:/opt/rocm-6.2.0/lib/llvm/bin/clang++  --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a  -std=c++14 -Wall -O3 -c -x hip main.cu -o "main.o"
make: *** [Makefile:55: main.o] Error 1
```

# lda-omp
## make -f Makefile.nvc
```
NVC++-W-0155-Compiler failed to translate accelerator region (see -Minfo messages): Missing branch target block (main.cpp: 59)
void EstepKernel<500, 256, 4000>(int const*, int const*, bool const*, float const*, bool, int, int, int, int, float const*, float const*, float*, float*, float*, float*, float*, int*):
(中略)
NVC++-F-0704-Compilation aborted due to previous errors. (main.cpp)
NVC++/arm Linux 24.9-0: compilation aborted
make: *** [Makefile.nvc:57: main.o] Error 2
```

# ldpc-omp
## make -f Makefile.aomp
```
In file included from main.cpp:16:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/math.h:20:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/cmath:86:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/__clang_hip_cmath.h:20:
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/type_traits:2360:21: error: static assertion failed due to requirement '__declval_protector<long>::__stop': declval() must not be used!
 2360 |       static_assert(__declval_protector<_Tp>::__stop,
      |
(中略)
2 warnings and 7 errors generated.
make: *** [Makefile.aomp:61: main.o] Error 1
```

# lebesgue-hipified
## make
```
/opt/rocm-6.2.0/include/hip/amd_detail/amd_hip_atomic.h:920:8: note: previous definition is here
  920 | double atomicMax(double* addr, double val) {
      |        ^
kernels.cu:8:8: error: redefinition of 'atomicMax'
    8 | double atomicMax(double *address, double val)
      |
(中略)
1 warning and 1 error generated when compiling for gfx90a.
failed to execute:/opt/rocm-6.2.0/lib/llvm/bin/clang++  --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a  -std=c++14 -Wall -O3 -c -x hip kernels.cu -o "kernels.o"
make: *** [Makefile:49: kernels.o] Error 1
```
# lebesgue-omp
## make -f Makefile.aomp
```
In file included from ../lebesgue-cuda/main.cpp:3:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/math.h:20:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/cmath:86:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/__clang_hip_cmath.h:20:
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/type_traits:2360:21: error: static assertion failed due to requirement '__declval_protector<long>::__stop': declval() must not be used!
 2360 |       static_assert(__declval_protector<_Tp>::__stop,
      |
(中略)
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/chrono:3225:14: note: while substituting deduced template arguments into function template 'duration' [with _Rep2 = long double, $1 = (no value)]
 3225 |     { return chrono::duration<long double, ratio<3600,1>>{__hours}; }
      |              ^
7 errors generated.
make: *** [Makefile.aomp:62: main.o] Error 1
```


# leukocyte-omp
## make -f Makefile.aomp
```
In file included from detect_main.c:2:
In file included from ./track_ellipse.h:4:
In file included from ./find_ellipse.h:7:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/math.h:20:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/cmath:86:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/__clang_hip_cmath.h:20:
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/type_traits:2360:21: error: static assertion failed due to requirement '__declval_protector<long>::__stop': declval() must not be used!
 2360 |       static_assert(__declval_protector<_Tp>::__stop,
      |
(中略)
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/chrono:1153:9: note: in instantiation of function template specialization 'std::chrono::time_point_cast<std::chrono::duration<long, std::ratio<1, 1000000000>>, std::chrono::system_clock, std::chrono::duration<long>>' requested here
 1153 |         return time_point_cast<system_clock::duration>
      |                ^
5 errors generated.
make: *** [Makefile.aomp:59: detect_main.o] Error 1
```

# libor-omp
## make -f Makefile.aomp
```
In file included from main.cpp:22:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/math.h:20:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/cmath:86:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/__clang_hip_cmath.h:20:
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/type_traits:2360:21: error: static assertion failed due to requirement '__declval_protector<long>::__stop': declval() must not be used!
 2360 |       static_assert(__declval_protector<_Tp>::__stop,
      |
(中略)
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/chrono:3225:14: note: while substituting deduced template arguments into function template 'duration' [with _Rep2 = long double, $1 = (no value)]
 3225 |     { return chrono::duration<long double, ratio<3600,1>>{__hours}; }
      |              ^
7 errors generated.
make: *** [Makefile.aomp:60: main.o] Error 1
```

# lid-driven-cavity-omp
## make -f Makefile.aomp
```
In file included from main.cpp:21:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/math.h:20:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/cmath:86:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/__clang_hip_cmath.h:20:
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/type_traits:2360:21: error: static assertion failed due to requirement '__declval_protector<long>::__stop': declval() must not be used!
 2360 |       static_assert(__declval_protector<_Tp>::__stop,
      |                     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(中略)
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/chrono:3225:14: note: while substituting deduced template arguments into function template 'duration' [with _Rep2 = long double, $1 = (no value)]
 3225 |     { return chrono::duration<long double, ratio<3600,1>>{__hours}; }
      |              ^
7 errors generated.
make: *** [Makefile.aomp:61: main.o] Error 1
```

# lif-omp

## make -f Makefile.aomp

```
In file included from main.cpp:3:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/math.h:20:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/cmath:86:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/__clang_hip_cmath.h:20:
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/type_traits:2360:21: error: static assertion failed due to requirement '__declval_protector<long>::__stop': declval() must not be used!
 2360 |       static_assert(__declval_protector<_Tp>::__stop,
      |
(中略)
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/chrono:3225:14: note: while substituting deduced template arguments into function template 'duration' [with _Rep2 = long double, $1 = (no value)]
 3225 |     { return chrono::duration<long double, ratio<3600,1>>{__hours}; }
      |              ^
7 errors generated.
make: *** [Makefile.aomp:60: main.o] Error 1
```

# linearprobing-omp
## make -f Makefile.nvc

```
nvc++  -Wall  -O3 -Minfo -mp=gpu -gpu=cc70 -c linearprobing.cpp -o linearprobing.o
hash(unsigned int):
     10, Generating implicit omp declare target routine
         Generating NVIDIA GPU code
NVC++-W-0155-Compiler failed to translate accelerator region (see -Minfo messages): Missing branch target block (linearprobing.cpp: 28)
insert_hashtable(KeyValue*, KeyValue const*, unsigned int):
     28, #omp target teams distribute parallel for num_teams(gridsize) thread_limit(256)
         28, Generating "nvkernel__Z16insert_hashtableP8KeyValuePKS_j_F1L28_2" GPU kernel
NVC++-F-0704-Compilation aborted due to previous errors. (linearprobing.cpp)
NVC++/arm Linux 24.9-0: compilation aborted
make: *** [Makefile.nvc:57: linearprobing.o] Error 2
```

# logic-resim-cuda
## make

```
src/sim/Simulator.cu(237): error: invalid narrowing conversion from "signed char" to "char"
          oPort[currSize] = {currTime, oVcurr};
                                       ^
(中略)
1 error detected in the compilation of "src/sim/Simulator.cu".
make: *** [Makefile:51: main] Error 2
```


# logic-rewrite-hip
## make

```
refactor_mffc.cu:247:20: error: use of undeclared identifier '__activemask'
  247 |         warpMask = __activemask();
(中略)
123 warnings and 3 errors generated when compiling for gfx90a.
failed to execute:/opt/rocm-6.2.0/lib/llvm/bin/clang++  --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a  -std=c++17 -Wall -O3 -ffast-math -c -x hip refactor_mffc.cu -o "refactor_mffc.o"
make: *** [Makefile:52: refactor_mffc.o] Error 1
```    
# logic-rewrite-hipified
## make

```
In file included from aig_manager.cu:11:
./common.h:201:58: error: no matching function for call to 'hipGetErrorString'
  201 |       printf("GPU kernel assert: %s,\nat %s, line %d\n", hipGetErrorString(code), file, line);
      |                                                          ^~~~~~~~~~~~~~~~~

(中略)
58 warnings and 1 error generated when compiling for gfx90a.
failed to execute:/opt/rocm-6.2.0/lib/llvm/bin/clang++  --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a  -std=c++17 -Wall -O3 -c -x hip aig_manager.cu -o "aig_manager.o"
make: *** [Makefile:53: aig_manager.o] Error 1
```

# logprob-hip
## make

```
hipcc  -std=c++14 -Wall -fopenmp -I../logprob-cuda/ -O3 -ffast-math -DWAVE64 -c main.cu -o main.o
In file included from main.cu:26:
../logprob-cuda/reference.h:1:20: error: redefinition of 'HALF_FLT_MAX'
    1 | static const float HALF_FLT_MAX = 65504.F;
      |                    ^
./reduce.h:1:20: note: previous definition is here
    1 | static const float HALF_FLT_MAX = 65504.F;
      |                    ^
1 error generated when compiling for gfx90a.
failed to execute:/opt/rocm-6.2.0/lib/llvm/bin/clang++  --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a  -std=c++14 -Wall -fopenmp -I../logprob-cuda/ -O3 -ffast-math -DWAVE64 -c -x hip main.cu -o "main.o"
make: *** [Makefile:54: main.o] Error 1
```

# logprob-hipified
## make

```
hipcc  -std=c++14  -Wall -fopenmp -O3  -c main.cu -o main.o
In file included from main.cu:26:
./reduce.h:9:12: error: use of undeclared identifier '__shfl_xor_sync'
    9 |     val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
      |
(中略)
2 errors generated when compiling for gfx90a.
failed to execute:/opt/rocm-6.2.0/lib/llvm/bin/clang++  --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a  -std=c++14 -Wall -fopenmp -O3 -c -x hip main.cu -o "main.o"
make: *** [Makefile:51: main.o] Error 1
```

# lombscargle-omp
## make -f Makefile.aomp

```
In file included from main.cpp:37:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/math.h:20:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/cmath:86:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/__clang_hip_cmath.h:20:
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/type_traits:2360:21: error: static assertion failed due to requirement '__declval_protector<long>::__stop': declval() must not be used!
 2360 |       static_assert(__declval_protector<_Tp>::__stop,
      |
(中略)
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/chrono:3225:14: note: while substituting deduced template arguments into function template 'duration' [with _Rep2 = long double, $1 = (no value)]
 3225 |     { return chrono::duration<long double, ratio<3600,1>>{__hours}; }
      |              ^
7 errors generated.
make: *** [Makefile.aomp:60: main.o] Error 1
```

# lr-omp
## make -f Makefile.aomp

```
In file included from linear_par.cpp:1:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/math.h:20:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/cmath:86:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/__clang_hip_cmath.h:20:
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/type_traits:2360:21: error: static assertion failed due to requirement '__declval_protector<long>::__stop': declval() must not be used!
 2360 |       static_assert(__declval_protector<_Tp>::__stop,
      |
(中略)
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/chrono:3225:14: note: while substituting deduced template arguments into function template 'duration' [with _Rep2 = long double, $1 = (no value)]
 3225 |     { return chrono::duration<long double, ratio<3600,1>>{__hours}; }
      |              ^
7 errors generated.
make: *** [Makefile.aomp:61: linear_par.o] Error 1
```

# lsqt-cuda
## make -f Makefile.gpu

```
nvcc -O3 -std=c++11 -arch=sm_60 -use_fast_math -DDEBUG -c vector.cu -o obj_gpu/vector.o
vector.cu(251): error: identifier "__syncwrap" is undefined
    v += s[t + 32]; __syncwrap();
                    ^
1 error detected in the compilation of "vector.cu".
make: *** [Makefile.gpu:23: obj_gpu/vector.o] Error 2
```
# lsqt-cuda
## make -f Makefile.gpu
```
hipcc -O3 -std=c++11   -DDEBUG -c vector.cu -o obj_gpu/vector.o
vector.cu:252:21: error: use of undeclared identifier '__syncwrap'; did you mean '__sync_swap'?
  252 |   v += s[t + 32];   __syncwrap();
      |                     ^~~~~~~~~~
      |                     __sync_swap
(中略)
fatal error: too many errors emitted, stopping now [-ferror-limit=]
20 errors generated when compiling for gfx90a.
failed to execute:/opt/rocm-6.2.0/lib/llvm/bin/clang++  --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a  -O3 -std=c++11 -DDEBUG -c -x hip vector.cu -o "obj_gpu/vector.o"
make: *** [Makefile.gpu:23: obj_gpu/vector.o] Error 1
```

# ludb-hipified

## make

```
hipcc  -std=c++14  -Wall  -O3 -c main.cu -o main.o
main.cu:49:10: fatal error: 'hipblas.h' file not found
   49 | #include <hipblas.h>
      |          ^~~~~~~~~~~
1 error generated when compiling for gfx90a.
failed to execute:/opt/rocm-6.2.0/lib/llvm/bin/clang++  --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a  -std=c++14 -Wall -O3 -c -x hip main.cu -o "main.o"
make: *** [Makefile:51: main.o] Error 1
```

# mallocFree-hipified

# make
```
hipcc  -std=c++14  -Wall  -O3 -c main.cu -o main.o
main.cu:61:5: error: no matching function for call to 'hipHostAlloc'
   61 |     hipHostAlloc(&Ad, size, hipHostMallocMapped);
      |     ^~~~~~~~~~~~
/opt/rocm-6.2.0/include/hip/hip_runtime_api.h:3780:12: note: candidate function not viable: no known conversion from 'int **' to 'void **' for 1st argument
 3780 | hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags);
      |            ^            ~~~~~~~~~~
main.cu:157:7: error: no matching function for call to 'hipHostAlloc'
  157 |       hipHostAlloc(&Ad[j], size[i], hipHostMallocMapped);
      |       ^~~~~~~~~~~~
/opt/rocm-6.2.0/include/hip/hip_runtime_api.h:3780:12: note: candidate function not viable: no known conversion from 'int **' to 'void **' for 1st argument
 3780 | hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags);
      |            ^            ~~~~~~~~~~
2 errors generated when compiling for gfx90a.
failed to execute:/opt/rocm-6.2.0/lib/llvm/bin/clang++  --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a  -std=c++14 -Wall -O3 -c -x hip main.cu -o "main.o"
make: *** [Makefile:50: main.o] Error 1
```

# mallocFree-omp
## make -f Makefile.nvc
```
nvc++  -Wall  -DUM -O3 -Minfo -mp=gpu -gpu=cc70 -c main.cpp -o main.o
NVC++-S-0155-OpenMP unified_shared_memory clause is used in the requires construct but the application is not compiled with -gpu=unified.  (main.cpp)
NVC++/arm Linux 24.9-0: compilation completed with severe errors
make: *** [Makefile.nvc:62: main.o] Error 2
```

# marchingCubes-hipified

## make
```
hipcc  -std=c++14  -Wall  -O3 -c main.cu -o main.o
main.cu:81:10: error: use of undeclared identifier '__shfl_down_sync'; did you mean '__shfl_down'?
   81 |     t0 = __shfl_down_sync(0xffffffffu, minV, c0);
      |          ^~~~~~~~~~~~~~~~
      |          __shfl_down
(中略)
main.cu:368:24: error: use of undeclared identifier '__shfl_up_sync'
  368 |       unsigned int tp1(__shfl_up_sync(0xffffffffu, warpSumTriangles, c0));
      |                        ^
14 errors generated when compiling for gfx90a.
failed to execute:/opt/rocm-6.2.0/lib/llvm/bin/clang++  --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a  -std=c++14 -Wall -O3 -c -x hip main.cu -o "main.o"
make: *** [Makefile:52: main.o] Error 1
```

# match-omp
## make -f Makefile.aomp

```
In file included from main.cpp:9:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/cmath:
86:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/__clang_hip_cmath.h:20
:
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/type_traits:2360:21: error: static assertion failed due to requirement '__declval_protector<long>::__stop': declval() must not be used!
 2360 |       static_assert(__declval_protector<_Tp>::__stop,
      |
(中略)
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/bits/basic_string.h:6725:23: note: in instantiation of function template specialization '__gnu_cxx::__to_xstring<std::basic_string<char>, char>' requested here
 6725 |     return __gnu_cxx::__to_xstring<string>(&std::vsnprintf, __n,
      |                       ^
2 warnings and 8 errors generated.
make: *** [Makefile.aomp:61: main.o] Error 1
```

#  matern-omp

## make -f Makefile.aomp

```
In file included from main.cpp:3:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/math.h:20:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/cmath:86:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/__clang_hip_cmath.h:20:
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/type_traits:2360:21: error: static assertion failed due to requirement '__declval_protector<long>::__stop': declval() must not be used!
 2360 |       static_assert(__declval_protector<_Tp>::__stop,
(中略)
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/chrono:3225:14: note: while substituting deduced template arguments into function template 'duration' [with _Rep2 = long double, $1 = (no value)]
 3225 |     { return chrono::duration<long double, ratio<3600,1>>{__hours}; }
      |              ^
```

# matrix-rotate-omp

## make -f Makefile.aomp
```
In file included from main.cpp:1:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/cmath:86:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/__clang_hip_cmath.h:20:
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/type_traits:2360:21: error: static assertion failed due to requirement '__declval_protector<long>::__stop': declval() must not be used!
 2360 |       static_assert(__declval_protector<_Tp>::__stop,
(中略)
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/chrono:3225:14: note: while substituting deduced template arguments into function template 'duration' [with _Rep2 = long double, $1 = (no value)]
 3225 |     { return chrono::duration<long double, ratio<3600,1>>{__hours}; }
      |              ^
7 errors generated.
make: *** [Makefile.aomp:60: main.o] Error 1
```

# maxpool3d-omp

## make -f Makefile.aomp
```
In file included from main.cpp:4:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/math.h:20:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/cmath:86:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/__clang_hip_cmath.h:20:
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/type_traits:2360:21: error: static assertion failed due to requirement '__declval_protector<long>::__stop': declval() must not be used!
 2360 |       static_assert(__declval_protector<_Tp>::__stop,
      |                     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(中略)
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/chrono:3225:14: note: while substituting deduced template arguments into function template 'duration' [with _Rep2 = long double, $1 = (no value)]
 3225 |     { return chrono::duration<long double, ratio<3600,1>>{__hours}; }
      |              ^
7 errors generated.
make: *** [Makefile.aomp:60: main.o] Error 1
```

# mcmd-omp

## make -f Makefile.aomp
```
In file included from ../mcmd-cuda/main.cpp:9:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/cmath:
86:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/__clang_hip_cmath.h:20
:
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/type_traits:2360:21: error: static assertion failed due to requirement '__declval_protector<_IO_FILE *&>::__stop': declval() must not be used!
 2360 |       static_assert(__declval_protector<_Tp>::__stop,
(中略)
../mcmd-cuda/classes.cpp:1093:12: note: in instantiation of member function 'std::vector<Constants::UniqueAngle>::~vector' requested here
 1093 | Constants::Constants() {
      |            ^
fatal error: too many errors emitted, stopping now [-ferror-limit=]
5 warnings and 20 errors generated.
make: *** [Makefile.aomp:60: main.o] Error 1
```

# mcpr-omp

## make -f Makefile.aomp
```
n file included from main.cpp:4:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/cmath:86:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/__clang_hip_cmath.h:20:
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/type_traits:2360:21: error: static assertion failed due to requirement '__declval_protector<unsigned int *>::__stop': declval() must not be used!
 2360 |       static_assert(__declval_protector<_Tp>::__stop,
      |                     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(中略)
ain.cpp:58:16: note: in instantiation of member function 'std::mersenne_twister_engine<unsigned long, 32, 624, 397, 31, 2567483615, 11, 4294967295, 7, 2636928640, 15, 4022730752, 18, 1812433253>::mersenne_twister_engine' requested here
   58 |   std::mt19937 gen(19937);
      |                ^
12 errors generated.
make: *** [Makefile.aomp:60: main.o] Error 1
```

# md5hash-omp

## make -f Makefile.aomp

```
In file included from MD5Hash.cpp:1:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/math.h
:20:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/openmp_wrappers/cmath:
86:
In file included from /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include/__clang_hip_cmath.h:20
:
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/type_traits:2360:21: error:
 static assertion failed due to requirement '__declval_protector<_IO_FILE *&>::__stop': de
clval() must not be used!
 2360 |       static_assert(__declval_protector<_Tp>::__stop,
(中略)
/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/bits/basic_string.h:6725:23: note: in instantiation of function template specialization '__gnu_cxx::__to_xstring<std::basic_string<char>, char>' requested here
 6725 |     return __gnu_cxx::__to_xstring<string>(&std::vsnprintf, __n,
      |                       ^
10 errors generated.
make: *** [Makefile.aomp:62: MD5Hash.o] Error 1
```

# mdh-hipified
## make
```
In file included from main.cu:22:
./helper_math.h:42:10: fatal error: 'cuda_runtime.h' file not found
   42 | #include "cuda_runtime.h"
      |          ^~~~~~~~~~~~~~~~
1 error generated when compiling for gfx90a.
failed to execute:/opt/rocm-6.2.0/lib/llvm/bin/clang++  --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a  -std=c++14 -Wall -fopenmp -O3 -c -x hip main.cu -o "main.o"
make: *** [Makefile:52: main.o] Error 1
```
あるいは
```
hipcc  -std=c++14  -Wall  -fopenmp  -O3 -c main.cu -o main.o
In file included from main.cu:22:
./helper_math.h:1475:30: error: use of overloaded operator '/' is ambiguous (with operand types 'float2' (aka 'HIP_vector_type<float, 2>') and 'float2')
 1475 |     float2 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
(中略)
fatal error: too many errors emitted, stopping now [-ferror-limit=]
20 errors generated when compiling for gfx90a.
failed to execute:/opt/rocm-6.2.0/lib/llvm/bin/clang++  --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a --offload-arch=gfx90a  -std=c++14 -Wall -fopenmp -O3 -c -x hip main.cu -o "main.o"
make: *** [Makefile:52: main.o] Error 1
```

# memcpy-omp

## make -f Makefile.nvc
```
"main.cpp", line 72: error: expression must be a modifiable lvalue
        time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        ^
"main.cpp", line 79: error: identifier "timeH2D" is undefined
                << (float)std::abs(timeH2D - timeD2H) / (repeat * size[i]);
                                   ^
"main.cpp", line 79: error: identifier "timeD2H" is undefined
                << (float)std::abs(timeH2D - timeD2H) / (repeat * size[i]);
                                             ^
3 errors detected in the compilation of "main.cpp".
make: *** [Makefile.nvc:58: main.o] Error 2
```
## make -f Makefile.aomp
```
main.cpp:72:12: error: non-object type 'time_t (time_t *) noexcept(true)' (aka 'long (long *) noexcept(true)') is not assignable
   72 |       time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      |       ~~~~ ^
main.cpp:79:34: error: use of undeclared identifier 'timeH2D'
   79 |               << (float)std::abs(timeH2D - timeD2H) / (repeat * size[i]);
      |                                  ^
main.cpp:79:44: error: use of undeclared identifier 'timeD2H'
   79 |               << (float)std::abs(timeH2D - timeD2H) / (repeat * size[i]);
3 errors generated.
make: *** [Makefile.aomp:61: main.o] Error 1
```
