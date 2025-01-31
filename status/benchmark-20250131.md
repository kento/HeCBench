| name | cuda | hip | hipified | omp_nvc | omp_aomp |
| -- | -- | -- | -- | -- | -- |
| accuracy | 5.55 | 6.49 | 6.48 | 63.77 | 10.19 |
| ace | 9.52 | 3.35 | 3.32 | 5.15 | exe err |
| adam | 7.08 | 4.20 | 4.19 | 8.65 | exe err |
| addBiasResidualLayerNorm | 18.92 | 50.36 | exe err | -- | -- |
| adv | 7.32 | 12.87 | 13.06 | 7.81 | 4.22 |
| aes | 8.95 | 1.23 | 1.21 | 8.99 | 0.74 |
| affine | 7.76 | 4.76 | 4.77 | 14.42 | |
| aidw | 11.02 | 9.75 | 48.61 | 11.46 | 87.89 |
| aligned-types | 5.70 | 2.16 | 2.16 | 6.91 | 3.42 |
| all-pairs-distance | 6.19 | 22.04 | 21.94 | 20.97 | 3.41 |
| allreduce | | | | -- | -- |
| amgmk | 4.53 | 0.75 | 0.74 | 4.52 | 0.80 |
| ans | 7.25 | | | 6.10 | |
| aobench | 4.48 | 0.48 | 0.48 | 7.60 | 1.53 |
| aop | 8.77 | 34.59 | | 306.19 | 0.86 |
| asmooth | 18.28 | 12.71 | 13.61 | 13.83 | |
| assert | 4.36 | exe err | exe err | -- | -- |
| asta | 6.24 | 3.67 | 3.69 | 12.83 | 0.34 |
| atan2 | 5.30 | 0.76 | 0.76 | 5.37 | |
| atomicAggregate | 7.63 | 5.48 | | -- | -- |
| atomicCAS | 17.72 | 14.55 | 15.02 | -- | -- |
| atomicCost | 97.45 | 7.86 | 7.93 | 96.88 | 25.15 |
| atomicIntrinsics | 89.73 | 70.29 | 70.31 | 4.58 | 0.90 |
| atomicPerf | 11.11 | 5.26 | | 12.14 | 372.30 |
| atomicReduction | 5.63 | 0.92 | 0.92 | | 1.79 |
| atomicSystemWide | 5.31 | 10.19 | 11.58 | -- | -- |
| attention | 40.64 | 158.70 | | 0.01 | |
| attentionMultiHead | 5.36 | 1.66 | build err | -- | -- |
| axhelm | 5.01 | | 0.81 | 10.07 | 0.73 |
| b+tree | 4.89 | 0.96 | 0.94 | 5.40 | 0.72 |
| babelstream | 4.64 | 0.95 | 0.95 | 4.65 | 3.11 |
| background-subtract | 5.70 | 4.74 | 4.73 | 7.43 | |
| backprop | 4.51 | 0.46 | 0.46 | 4.94 | 0.25 |
| bezier-surface | 18.36 | 23.79 | 23.80 | 17.16 | |
| bfs | 5.32 | 1.18 | 1.12 | 5.31 | 0.95 |
| bh | 8.16 | | | -- | -- |
| bicgstab | 5.17 | | | -- | -- |
| bilateral | 9.52 | 46.72 | 18.22 | 10.11 | |
| bincount | 5.48 | 2.01 | 2.01 | -- | -- |
| binomial | 4.92 | 1.78 | 1.79 | -- | |
| bitcracker | exe err | exe err | exe err | -- | -- |
| bitonic-sort | 16.45 | 13.42 | 13.35 | 11.35 | |
| bitpacking | 8.14 | 4.10 | 4.09 | -- | -- |
| bitpermute | 6.01 | 7.11 | | -- | -- |
| black-scholes | 9.55 | 3.97 | 3.95 | 7.78 | 8.32 |
| blas-dot | 17.29 | 9.95 | | -- | -- |
| blas-fp8gemm | 4.52 | | | -- | -- |
| blas-gemm | 12.53 | 4.51 | | -- | -- |
| blas-gemmBatched | 19.85 | 21.61 | | -- | -- |
| blas-gemmEx | 15.99 | | | -- | -- |
| blas-gemmEx2 | 16.02 | | | -- | -- |
| blas-gemmStridedBatched | 20.18 | 21.54 | | -- | -- |
| blockAccess | 5.76 | 8.15 | 8.13 | -- | -- |
| blockexchange | 33.67 | | | -- | -- |
| bm3d | 0.32 | 0.06 | 0.04 | -- | -- |
| bmf | 5.99 | 7.26 | | -- | -- |
| bn | 6.33 | 10.18 | 10.23 | 6.72 | |
| bonds | 14.44 | 19.26 | 19.28 | 16.35 | 15.18 |
| boxfilter | 8.75 | 9.19 | | 11.78 | 37.43 |
| bscan | 10.62 | 1.78 | 0.44 | -- | -- |
| bsearch | 743.47 | 4.61 | 4.61 | 72.35 | 34.57 |
| bspline-vgh | 14.11 | 10.08 | 10.08 | 9.98 | 11.30 |
| bsw | 4.56 | 0.56 | | -- | -- |
| btree | 157.30 | -- | | -- | -- |
| burger | 110.83 | 21.49 | 21.50 | 75.83 | |
| bwt | 5904.43 | 13.38 | 13.66 | 5837.90 | 15.18 |
| car | 11.59 | 20.00 | 20.00 | 11.59 | |
| cbsfil | 43.47 | 5.94 | 5.94 | 42.67 | |
| cc | 3.49 | 2.48 | build err | -- | -- |
| ccl | | | | -- | -- |
| ccs | 5.47 | 6.11 | 6.10 | 7.24 | 0.27 |
| ccsd-trpdrv | 41.53 | 19.17 | 19.35 | 35.93 | 2.34 |
| ced | 7.05 | 3.23 | 3.21 | -- | -- |
| cfd | 5.10 | 1.51 | 1.48 | 5.44 | 9.98 |
| chacha20 | 6.78 | 8.25 | 8.24 | 8.89 | 7.34 |
| channelShuffle | 49.94 | 38.92 | 32.81 | 68.93 | 76.38 |
| channelSum | 66.15 | 52.19 | 52.10 | 69.99 | |
| che | 0.00 | 0.00 | 0.00 | | 0.00 |
| chemv | 5.08 | 1.90 | 1.91 | 5.54 | |
| chi2 | 223.36 | 20.43 | 20.42 | 260.91 | |
| clenergy | 4.83 | 1.34 | 1.34 | 5.26 | 17.58 |
| clink | 8.16 | 5.14 | 5.15 | 16.48 | 11.76 |
| clock | 4.48 | 0.44 | 0.42 | -- | -- |
| cm | 0.00 | 0.17 | 0.16 | | |
| cmembench | | | | -- | -- |
| cmp | | | | | |
| cobahh | 541.25 | | | | |
| collision | 9.90 | | | -- | -- |
| colorwheel | 48.47 | | | 42.53 | |
| columnarSolver | 11.55 | | | 13.73 | |
| complex | 25.25 | 3.68 | 3.67 | 15.22 | 26.06 |
| compute-score | 6.84 | 4.72 | 4.74 | 8.43 | |
| concat | 26.60 | 13.27 | 13.17 | 26.21 | 57.74 |
| concurrentKernels | 283.38 | 31.63 | 31.63 | -- | -- |
| contract | 1497.61 | 15.19 | 15.24 | 1418.37 | 24.22 |
| conversion | 6.77 | 0.94 | | 5.45 | 1.88 |
| convolution1D | 1715.16 | 100.98 | 100.98 | 3870.95 | |
| convolution3D | 2.86 | 83.03 | | 4.63 | |
| convolutionDeformable | | | | -- | -- |
| convolutionSeparable | 19.86 | 25.91 | 25.75 | 30.44 | |
| cooling | 2269.87 | 1.06 | 1.89 | 1115.12 | |
| coordinates | 510.74 | 3.35 | 3.34 | -- | -- |
| copy | 30.88 | 24.36 | | -- | -- |
| crc64 | 159.62 | 4.06 | 4.05 | 168.38 | 4.52 |
| cross | 12.96 | 11.32 | 11.43 | 387.76 | |
| crossEntropy | 11.71 | 10.41 | 10.40 | -- | -- |
| crs | 1259.70 | 10.30 | 10.30 | | 11.64 |
| d2q9-bgk | 5.74 | 1.48 | 1.48 | 7.58 | |
| d3q19-bgk | | 24.91 | 24.99 | -- | -- |
| damage | 25.55 | 1.02 | 1.02 | 222.52 | 0.79 |
| daphne | | | | -- | -- |
| dct8x8 | 181.81 | 1.50 | 1.51 | exe err | |
| ddbp | 18.57 | 4.88 | 4.88 | 18.44 | |
| debayer | 15.96 | 0.93 | 1.08 | exe err | 1.39 |
| degrid | 17.95 | 48.54 | | 22.38 | 115.32 |
| dense-embedding | 198.45 | 179.92 | 179.99 | 1777.93 | |
| depixel | 1163.31 | 30.94 | 37.48 | 6489.09 | |
| deredundancy | 18.36 | 59.29 | 59.26 | 18.47 | 13.02 |
| determinant | 4.73 | | | -- | -- |
| diamond | exe err | | | exe err | |
| dispatch | 6.37 | 1.98 | 2.02 | -- | -- |
| distort | 4.51 | 0.52 | 0.51 | 4.57 | 1.52 |
| divergence | 6.50 | 1.64 | 1.64 | 5.03 | 4.81 |
| doh | 18.82 | 19.75 | 19.75 | 6.18 | |
| dp | 54.96 | | | 34.07 | 31.61 |
| dpid | 50.59 | 1.02 | | -- | -- |
| dropout | 31.01 | 0.93 | 0.93 | -- | -- |
| dslash | 6.75 | 2.19 | 4.42 | 15.43 | 4.68 |
| dwconv | 81.64 | 23.33 | 23.22 | -- | -- |
| dwconv1d | | | | -- | -- |
| dxtc1 | 4.42 | 0.62 | 0.63 | -- | -- |
| dxtc2 | 4.53 | 0.66 | | 6.69 | 0.60 |
| easyWave | 7.11 | 2.48 | 2.46 | 7.25 | 3.17 |
| ecdh | 37.30 | 2.97 | 2.98 | 40.94 | 43.45 |
| egs | 5.07 | 0.17 | | -- | -- |
| eigenvalue | 23.90 | 13.44 | 13.40 | | 16.98 |
| eikonal | 206.17 | 10.50 | 9.93 | -- | -- |
| entropy | 57.12 | 10.30 | 10.31 | 57.16 | |
| epistasis | 67.92 | 189.71 | 190.18 | 136.80 | |
| ert | 7.38 | 3.53 | 3.53 | -- | -- |
| expdist | 9.41 | 7.22 | 7.25 | 15.58 | |
| extend2 | 7.56 | 11.35 | 11.35 | 6.81 | |
| extrema | 8.83 | 6.38 | 6.40 | 9.40 | 20.70 |
| f16max | 25.83 | 35.51 | 35.49 | -- | -- |
| f16sp | 33.71 | 30.75 | | -- | -- |
| face | 5.04 | 1.05 | 1.05 | | 0.69 |
| fdtd3d | 35.55 | 4.92 | 4.67 | 49.94 | 4.43 |
| feynman-kac | 84.36 | 113.98 | 115.15 | 24.12 | |
| fft | 6.52 | 2.18 | 2.18 | 7.49 | 0.49 |
| fhd | 765.79 | 1.11 | 1.09 | | |
| filter | 174.29 | 15.03 | 15.04 | 174.81 | 7.50 |
| flame | 5.83 | 3.90 | 3.93 | -- | -- |
| flip | 85.49 | 10.28 | 10.30 | 90.87 | 43.39 |
| floydwarshall | 78.88 | 1.92 | 1.95 | 97.80 | 5.67 |
| floydwarshall2 | 10.25 | 6.22 | | -- | -- |
| fluidSim | 22.99 | 25.92 | 25.54 | 21.98 | 30.36 |
| fpc | 70.35 | 1.40 | 1.40 | 30.19 | 0.80 |
| fpdc | 7.70 | 2.98 | 2.96 | | 2.19 |
| frechet | 15.26 | 1.11 | 1.11 | 144.67 | |
| fresnel | 8.65 | 6.64 | | | |
| frna | 0.00 | | | 0.00 | |
| fsm | 5.75 | 6.99 | 7.00 | | 0.34 |
| fwt | 5.36 | | 1.61 | 5.83 | |
| ga | 52.84 | 5.01 | 5.02 | 36.55 | 5.93 |
| gabor | 120.39 | 4.97 | 8.04 | exe err | |
| gamma-correction | 2178.54 | 19.22 | 19.21 | 3350.52 | 16.87 |
| gaussian | 20.74 | 5.89 | 5.67 | 9.21 | |
| gc | 4.56 | 0.51 | | | 0.14 |
| gd | 13.68 | 12.99 | 28.99 | 14.83 | 12.90 |
| ge-spmm | 7.54 | 8.63 | 8.59 | -- | -- |
| geam | 127.40 | | | -- | -- |
| gels | 4.67 | 1.83 | | -- | -- |
| gelu | 57.17 | 58.40 | 57.65 | -- | -- |
| gemv | 9.81 | 78.16 | | -- | -- |
| geodesic | exe err | exe err | exe err | exe err | exe err |
| gerbil | | | | -- | -- |
| gibbs | 4.53 | 0.71 | 0.70 | -- | -- |
| glu | 156.41 | 47.28 | 118.35 | 1593.95 | |
| gmm | 323.54 | 1.92 | 1.03 | 341.15 | |
| goulash | 142.59 | 11.60 | 11.66 | 147.07 | 20.07 |
| gpp | 6.83 | 238.43 | 234.16 | 6.99 | 29.99 |
| graphB+ | 7.21 | | | -- | -- |
| graphExecution | 6.60 | 1.84 | 1.84 | -- | -- |
| grep | 992.41 | 96.57 | 96.60 | 507.36 | 30.55 |
| grrt | 17.00 | 55.86 | 52.29 | 15.40 | |
| gru | 26.44 | 48.42 | 48.44 | -- | -- |
| haccmk | 6.54 | 4.25 | 4.25 | 6.38 | 5.41 |
| halo-finder | 5.44 | 4.50 | 7.77 | -- | -- |
| hausdorff | 32.73 | 14.40 | 14.40 | 33.92 | 15.64 |
| haversine | 6.46 | 1.82 | 2.05 | 6.52 | 2.09 |
| hbc | 15.68 | 15.25 | 15.24 | -- | -- |
| heartwall | 4.66 | 1.97 | 1.93 | exe err | 2.28 |
| heat | 26.31 | 39.72 | 40.25 | 91.48 | 69.52 |
| heat2d | build err | 11.03 | 11.50 | 248.52 | 329.62 |
| hellinger | 7.13 | 6.61 | 6.62 | 6.87 | 6.52 |
| henry | 5.87 | 2.89 | 4.51 | 5.07 | 3.89 |
| hexciton | 6.06 | 7.37 | build err | 6.82 | 17.83 |
| histogram | 10.43 | 1.30 | 1.33 | 5.30 | 8.38 |
| hmm | 7.18 | 3.94 | 3.97 | 17.31 | 7.21 |
| hogbom | 5.19 | 1.03 | 0.93 | 5.20 | 0.99 |
| hotspot | 4.71 | 0.60 | 0.59 | -- | -- |
| hotspot3D | 22.97 | 23.21 | 23.24 | 19.42 | 27.55 |
| hpl | 115.39 | 112.16 | build err | -- | -- |
| hungarian | 4.65 | 0.76 | 0.75 | -- | -- |
| hwt1d | 6.12 | 3.39 | 3.38 | 6.37 | 3.49 |
| hybridsort | 17.40 | 16.39 | 16.36 | 18.47 | 16.45 |
| hypterm | 10.92 | 15.09 | 15.05 | 18.92 | 20.58 |
| idivide | 7.92 | 6.23 | 6.23 | 11.15 | exe err |
| interleave | 6.02 | 3.46 | 3.47 | 5.30 | 3.39 |
| interval | 13.69 | 46.97 | 47.13 | 17.43 | 55.60 |
| intrinsics-cast | 69.54 | 9.21 | 9.22 | -- | -- |
| intrinsics-simd | 27.93 | -- | build err | -- | -- |
| inversek2j | 4.57 | 0.60 | 0.60 | 11.78 | 24.34 |
| is | 4.55 | 1.41 | 1.42 | -- | -- |
| ising | 33.95 | 15.69 | 15.71 | 33.13 | 22.85 |
| iso2dfd | 22.57 | 61.89 | 60.41 | 21.23 | 61.67 |
| jaccard | 197.46 | 254.66 | build err | -- | -- |
| jacobi | 4.81 | 8.91 | build err | 5.41 | 5.95 |
| jenkins-hash | 9.50 | 5.10 | 5.11 | 9.52 | 5.57 |
| kalman | 22.24 | 27.92 | 27.93 | 17.54 | 10.03 |
| keccaktreehash | 11.89 | 10.72 | 10.73 | 12.56 | 10.70 |
| keogh | 215.35 | 85.25 | 85.27 | 50.03 | 87.04 |
| kernelLaunch | 16.18 | 19.08 | 19.23 | 75.33 | 104.50 |
| kmc | 7.45 | 6.16 | 6.19 | -- | -- |
| kmeans | 32.53 | 42.64 | 42.41 | 30.35 | 49.75 |
| knn | 8.54 | 10.95 | 10.88 | 8.96 | 11.38 |
| kurtosis | 22.01 | 167.60 | 167.91 | -- | -- |
| lanczos | 6.90 | 3.34 | 3.26 | 7.31 | 3.91 |
| langevin | 73.86 | 10.60 | 12.36 | 74.80 | 16.99 |
| langford | 4.89 | 2.58 | 2.59 | build err | exe err |
| laplace | 5.84 | 4.45 | 4.45 | 9.40 | 21.52 |
| laplace3d | 95.07 | 18.61 | 17.75 | 56.92 | 22.79 |
| lavaMD | 104.12 | 61.88 | 61.77 | 194.21 | 55.93 |
| layernorm | 9.80 | 4.11 | build err | -- | -- |
| layout | 5.09 | 1.39 | 1.39 | 4.86 | 1.63 |
| lci | 19.97 | 8.41 | 8.41 | 4.37 | 7.82 |
| lda | 5.36 | 2.07 | 2.07 | build err | 14.90 |
| ldpc | 7.08 | 5.98 | 5.98 | 12.12 | 17.04 |
| lebesgue | 8.34 | 35.07 | | 6.62 | 36.22 |
| leukocyte | 4.76 | 1.16 | 1.02 | 6.07 | 1.15 |
| lfib4 | 188.11 | 31.34 | 28.44 | -- | -- |
| libor | 4.75 | 1.28 | 1.28 | 5.20 | 1.74 |
| lid-driven-cavity | 12.87 | 14.53 | 14.57 | 26.01 | 48.17 |
| lif | 133.75 | 76.25 | 76.29 | 150.82 | 80.63 |
| linearprobing | 106.78 | 76.77 | 76.54 | build err | exe err |
| log2 | 6.85 | 1.27 | 1.28 | exe err | 1.78 |
| logan | 10.75 | 15.29 | 15.64 | -- | -- |
| logic-resim | build err | 7.53 | 7.52 | -- | -- |
| logic-rewrite | 52.42 | build err | build err | -- | -- |
| logprob | 18.96 | 16.59 | build err | -- | -- |
| lombscargle | 4.83 | 1.18 | 1.18 | 4.88 | 1.18 |
| loopback | 7.82 | 10.90 | 10.88 | 10.54 | 11.34 |
| lr | 8.53 | 6.75 | 6.73 | 33.47 | 48.47 |
| lrn | 121.77 | 49.70 | 50.15 | 127.24 | 63.45 |
| lsqt | build err | 60.46 | build err | 59.87 | 74.43 |
| lud | 20.77 | 10.18 | 10.19 | 62.18 | 21.33 |
| ludb | 6.93 | 9.73 | build err | -- | -- |
| lulesh | 13.37 | 15.84 | 13.93 | 14.09 | 566.69 |
| lzss | | | | -- | -- |
| mallocFree | 5.60 | 9.40 | build err | 4.62 | 0.72 |
| mandelbrot | 7.37 | 16.41 | 15.85 | 5.63 | 13.34 |
| marchingCubes | 12.32 | 7.75 | build err | -- | -- |
| mask | 168.00 | 90.46 | 90.48 | 242.76 | 93.65 |
| match | 42.52 | 69.92 | 69.90 | 46.11 | 80.30 |
| matern | 17.87 | 31.70 | 31.69 | 116.98 | 111.40 |
| matrix-rotate | 30.57 | 9.78 | 10.18 | 31.16 | 10.20 |
| matrixT | 59.14 | 22.03 | 22.05 | -- | -- |
| maxFlops | 27.61 | 41.98 | 41.97 | 27.71 | 42.92 |
| maxpool3d | 40.06 | 15.72 | 14.82 | 40.38 | 15.40 |
| mcmd | 14.42 | 110.17 | 110.50 | 16.27 | 110.62 |
| mcpr | 12.75 | 60.69 | 60.66 | 14.44 | 67.98 |
| md | 18.53 | 19.47 | 19.51 | 18.97 | 18.84 |
| md5hash | 17.24 | 33.47 | 33.46 | 17.22 | 41.39 |
| mdh | 87.25 | 217.07 | build err | 40.38 | 218.96 |
| meanshift | 5.64 | 3.15 | 3.38 | 6.88 | 3.04 |
| medianfilter | 6.10 | 15.03 | 15.32 | 7.23 | 4.27 |
| memcpy | 7.45 | 19.46 | 19.69 | 8.45 | 18.21 |
| memtest | 20.17 | 28.04 | 28.04 | 21.57 | 38.18 |
| merge | 1657.99 | 1798.11 | 1799.55 | 1623.48 | 1814.24 |
| merkle | | 15.66 | build err | -- | -- |
| metropolis | 1511.38 | 71.73 | build err | | |
| mf-sgd | | | | -- | -- |
| michalewicz | 46.23 | 130.13 | 130.26 | | build err |
| miniDGS | | -- | | -- | -- |
| miniFE | | | | | |
| miniWeather | | build err | build err | | 127.07 |
| minibude | 5.27 | 2.47 | 2.47 | | build err |
| minimap2 | | build err | 1.87 | | build err |
| minimod | 91.44 | 2.85 | 2.86 | -- | -- |
| minisweep | 13.87 | 91.70 | 92.92 | | 2.97 |
| minkowski | 24.62 | 24.16 | 31.66 | | 89.66 |
| minmax | 273.04 | 4.51 | build err | -- | -- |
| mis | 94.24 | 0.47 | 0.47 | | |
| mixbench | 9.58 | build err | 50.78 | | build err |
| mmcsf | 8.26 | 8.10 | 8.14 | -- | -- |
| mnist | | 76.45 | 75.03 | -- | -- |
| morphology | 30.96 | 3.10 | 3.10 | | 0.92 |
| mpc | 4.55 | build err | build err | -- | -- |
| mr | 3.39 | 1.28 | 1.28 | | 2.39 |
| mrc | 117.28 | 4.17 | 4.12 | | build err |
| mrg32k3a | 241.19 | 0.84 | build err | -- | -- |
| mriQ | 6.97 | 6.90 | 28.14 | | build err |
| mt | | build err | build err | | build err |
| mtf | 66.29 | 20.05 | 20.00 | -- | -- |
| multimaterial | 30.05 | 28.64 | 28.69 | | build err |
| multinomial | 5.06 | 1.73 | 1.74 | -- | -- |
| murmurhash3 | 2.42 | 1.09 | 1.10 | | 1.87 |
| myocyte | 377.67 | 0.97 | 0.95 | | build err |
| nbnxm | | 8.56 | | -- | -- |
| nbody | | 0.49 | 0.49 | | 0.70 |
| ne | 8.15 | 2.30 | 2.27 | | build err |
| nlll | 48.67 | 149.47 | 149.47 | | build err |
| nms | | 0.61 | 0.60 | | build err |
| nn | 4.60 | 0.48 | 0.47 | | 0.73 |
| nonzero | 283.00 | 32.19 | 32.14 | -- | -- |
| norm2 | | build err | build err | | build err |
| nosync | exe err | 5.45 | build err | -- | -- |
| nqueen | 0.90 | 37.42 | 37.24 | | 28.63 |
| ntt | | 7.76 | 7.74 | | 10.11 |
| nw | 0.00 | 2.66 | 2.64 | | 2.96 |
| openmp | 16.40 | 66.84 | 66.87 | | 77.52 |
| opticalFlow | 5.92 | 3.05 | build err | -- | -- |
| overlap | | 0.56 | build err | -- | -- |
| overlay | 180.92 | 4.38 | 4.27 | | build err |
| p2p | | 0.78 | 0.81 | -- | -- |
| p4 | | 7.95 | 8.24 | | 3.29 |
| pad | 290.47 | 1.97 | build err | -- | -- |
| page-rank | 25.87 | 9.10 | 9.37 | | build err |
| particle-diffusion | | 9.68 | 9.66 | | 13.59 |
| particlefilter | 5.61 | 2.41 | 2.39 | | 4.97 |
| particles | | 1.99 | 1.99 | | build err |
| pathfinder | 54.42 | 1.35 | 1.35 | | 0.97 |
| pcc | 232.96 | build err | build err | -- | -- |
| perlin | | 13.10 | 13.06 | -- | -- |
| permutate | 27.69 | 34.46 | 34.64 | | 35.64 |
| permute | 120.73 | 1.06 | build err | | 1.78 |
| perplexity | 298.50 | 2.46 | 2.43 | | build err |
| phmm | | 93.71 | 93.71 | | 19.23 |
| pingpong | | build err | build err | -- | -- |
| pitch | | 4.11 | 4.11 | -- | -- |
| pnpoly | | 21.00 | 20.99 | | build err |
| pns | 34.52 | 7.48 | 7.48 | | 0.23 |
| pointwise | 92.62 | 1.96 | 3.06 | | build err |
| pool | 37.15 | 11.83 | 11.83 | | 21.57 |
| popcount | 939.39 | 4.31 | 4.30 | | 13.34 |
| prefetch | | 71.78 | 227.44 | -- | -- |
| present | | 4.64 | 4.64 | | 4.65 |
| prna | 0.00 | build err | build err | | |
| projectile | | 1.39 | 1.09 | | 4.29 |
| pso | 153.40 | 1.39 | 1.37 | | 1.92 |
| qem | | 11.34 | build err | -- | -- |
| qkv | | 13.80 | build err | -- | -- |
| qrg | | 20.09 | 20.10 | | build err |
| qtclustering | 2.96 | 0.94 | 0.94 | | build err |
| quicksort | | 12.10 | 12.03 | | build err |
| radixsort | | 1.66 | 1.67 | | build err |
| radixsort2 | 23.21 | 101.95 | 101.90 | -- | -- |
| rainflow | 42.57 | 7.21 | 7.21 | | 3.84 |
| randomAccess | | 14.06 | 13.91 | | 9.01 |
| rayleighBenardConvection | | 46.63 | build err | -- | -- |
| reaction | | 5.94 | 5.95 | | 22.70 |
| recursiveGaussian | | 4.35 | build err | | build err |
| relu | 1055.69 | 18.84 | build err | -- | -- |
| remap | | 22.12 | 21.89 | -- | -- |
| resize | 209.55 | 7.80 | 7.80 | | build err |
| resnet-kernels | 27.63 | 3.98 | 1.03 | -- | -- |
| reverse | | 1.72 | 1.71 | | build err |
| reverse2D | 35.86 | 1.45 | 2.66 | -- | -- |
| rfs | 358.55 | 13.29 | 13.14 | | build err |
| ring | 255.18 | 5.73 | 5.72 | -- | -- |
| rle | 173.36 | 0.68 | build err | -- | -- |
| rng-wallace | | 1.61 | 1.60 | | build err |
| rodrigues | 397.51 | 0.95 | 0.95 | | 5.43 |
| romberg | 91.07 | 1.13 | 1.10 | | build err |
| rotary | | 0.75 | build err | -- | -- |
| rowwiseMoments | | 2.43 | 2.43 | -- | -- |
| rsbench | | | | | 2.70 |
| rsc | 76.73 | 0.83 | 0.81 | | build err |
| rsmt | 5.39 | -- | | -- | -- |
| rtm8 | | 3.83 | 3.96 | | 4.58 |
| rushlarsen | 681.18 | 11.16 | 11.08 | | 11.25 |
| s3d | 2.93 | 0.64 | 0.62 | | 3.42 |
| s8n | 1236.81 | 28.46 | 28.40 | | 56.50 |
| sa | | 1.68 | 1.68 | -- | -- |
| sad | 4.78 | 2.08 | | | 8.00 |
| sampling | | 7.76 | 7.75 | | 2.15 |
| saxpy-ompt | | build err | build err | -- | -- |
| sc | | 1.14 | build err | -- | -- |
| scan | 1054.96 | 111.13 | 110.57 | | 104.33 |
| scan2 | 20.97 | 1.37 | 1.38 | | 4.09 |
| scan3 | 18.17 | 1.30 | 1.27 | -- | -- |
| scel | 22.06 | 12.28 | 23.28 | | build err |
| score | | 4.75 | build err | -- | -- |
| sddmm-batch | 206.93 | 212.57 | build err | -- | -- |
| seam-carving | 2.73 | 0.48 | | -- | -- |
| secp256k1 | | 1.28 | 1.28 | | 0.18 |
| segment-reduce | 560.80 | 8.57 | 8.60 | -- | -- |
| segsort | | 6.37 | build err | -- | -- |
| sheath | | 4.93 | 486.06 | | build err |
| shmembench | | 5.68 | 5.68 | | 1.11 |
| shuffle | | 7.95 | build err | -- | -- |
| si | | | | -- | -- |
| simpleMultiDevice | | 6.78 | 6.82 | -- | -- |
| simpleSpmv | 2123.72 | 13.45 | build err | | 10.00 |
| simplemoc | 213.07 | 4.38 | 4.30 | | 1261.04 |
| slit | | | | -- | -- |
| slu | | build err | | | build err |
| snake | 10.12 | 13.53 | 13.51 | 12.74 | 28.38 |
| sobel | 4.86 | 1.32 | 1.36 | 5.67 | 24.62 |
| sobol | | 3.27 | 3.27 | | 3.07 |
| softmax | | build err | build err | | 2.64 |
| softmax-fused | | 9.19 | build err | -- | -- |
| softmax-online | | 20.58 | build err | -- | -- |
| sort | | 6.06 | 6.05 | | build err |
| sortKV | | 39.88 | 39.58 | -- | -- |
| sosfil | | 6.63 | 6.62 | | build err |
| sparkler | | build err | build err | -- | -- |
| spaxpby | | build err | build err | -- | -- |
| spd2s | | 111.03 | build err | -- | -- |
| spgeam | | 19.13 | build err | -- | -- |
| spgemm | | 16.67 | build err | -- | -- |
| sph | | 3.06 | 3.06 | | build err |
| split | | 1.25 | 1.25 | | build err |
| spm | | 2.14 | 2.14 | | build err |
| spmm | | 7.19 | build err | -- | -- |
| spmv | | 6.13 | build err | -- | -- |
| spnnz | | 107.65 | build err | -- | -- |
| sps2d | | 118.14 | build err | -- | -- |
| spsm | | 117.88 | build err | -- | -- |
| spsort | | 104.18 | build err | -- | -- |
| sptrsv | 4.83 | 1.67 | 1.70 | | build err |
| srad | 4.51 | 0.54 | 0.54 | | build err |
| ss | 9.15 | 6.31 | 6.31 | | 922.90 |
| ssim | | 6.72 | build err | -- | -- |
| sss | 34.06 | build err | build err | -- | -- |
| sssp | 11.50 | 13.96 | 14.11 | -- | -- |
| stddev | | 24.36 | 33.47 | | build err |
| stencil1d | | 1.99 | 1.99 | | |
| stencil3d | | 6.76 | 6.70 | | 1.32 |
| streamCreateCopyDestroy | | 17.13 | 17.18 | -- | -- |
| streamOrderedAllocation | | 18.78 | 19.47 | -- | -- |
| streamPriority | | 2.84 | 1.44 | -- | -- |
| streamUM | | 24.85 | build err | -- | -- |
| streamcluster | | 4.05 | 4.03 | | build err |
| stsg | | build err | build err | -- | -- |
| su3 | | 2.28 | 2.28 | | build err |
| surfel | | 10.46 | 10.46 | | 179.57 |
| svd3x3 | 6.47 | 2.76 | 2.86 | 6.41 | 10.93 |
| sw4ck | | 25.07 | 24.92 | | 193.21 |
| swish | | 2.86 | 2.15 | | build err |
| tensorAccessor | | 17.51 | 17.51 | -- | -- |
| tensorT | | 1.81 | 1.82 | | 7.44 |
| testSNAP | 12.34 | 5.85 | 5.72 | 11.41 | 0.50 |
| thomas | | 17.48 | 17.45 | | 23.93 |
| threadfence | | 0.87 | 0.85 | | build err |
| tissue | | 21.07 | 21.06 | | build err |
| tonemapping | | 11.44 | 11.67 | | 3.98 |
| tpacf | 13.75 | 15.83 | build err | -- | -- |
| tqs | | 1.64 | 1.66 | | build err |
| triad | | 1.42 | 1.42 | | build err |
| tridiagonal | | 27.51 | 27.29 | | build err |
| tsa | | 1.95 | 1.78 | | build err |
| tsne | | build err | build err | -- | -- |
| tsp | 6.97 | 10.86 | 14.42 | | build err |
| unfold | | 0.62 | build err | -- | -- |
| urng | 4.68 | 0.45 | 0.45 | 4.57 | 0.79 |
| vanGenuchten | | 5.64 | 5.62 | | build err |
| vmc | | 1.85 | 1.85 | | build err |
| vol2col | | 10.06 | 10.03 | | build err |
| vote | | 12.21 | build err | -- | -- |
| voxelization | 12.44 | 94.03 | 94.03 | -- | -- |
| warpexchange | 5.47 | 0.66 | build err | -- | -- |
| warpsort | | 1.20 | build err | -- | -- |
| wedford | | 15.38 | build err | -- | -- |
| winograd | | 0.96 | 0.96 | | 3.51 |
| wlcpow | | 8.22 | 7.90 | | build err |
| wmma | 18.97 | 5.04 | build err | -- | -- |
| word2vec | 15.05 | build err | build err | -- | -- |
| wordcount | | 9.01 | 9.01 | 21.18 | 8.53 |
| wsm5 | | 9.75 | 9.73 | | build err |
| wyllie | | 3.43 | 3.42 | | 3.09 |
| xlqc | | build err | build err | | build err |
| xsbench | 51.69 | 55.05 | 43.60 | | 27.78 |
| zerocopy | | 114.24 | 988.62 | -- | -- |
| zeropoint | | 1.96 | 1.97 | | build err |
| zmddft | | 3.77 | 3.78 | | 3.10 |
| zoom | | 8.17 | 8.14 | -- | -- |
