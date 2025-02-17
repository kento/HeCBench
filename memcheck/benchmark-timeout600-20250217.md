| name | cuda | hip | hipified | omp_nvc | omp_aomp |
| -- | -- | -- | -- | -- | -- |
| accuracy | 2.55 | 6.54 | build err | 59.31 | 8.41 |
| ace | 4.92 | 3.34 | 3.51 | 4.97 | build err |
| adam | 6.86 | 4.20 | 4.23 | 8.49 | build err |
| addBiasResidualLayerNorm | 16.70 | 50.37 | build err | -- | -- |
| adv | 4.71 | 12.90 | 13.12 | 6.43 | 4.06 |
| aes | 4.37 | 1.23 | 1.21 | 5.12 | 0.66 |
| affine | 7.53 | 4.75 | 4.77 | 14.21 | 51.65 |
| aidw | 6.17 | 9.72 | 48.22 | 6.40 | over 600 |
| aligned-types | 5.42 | 2.14 | 2.14 | 6.57 | 33.16 |
| all-pairs-distance | 5.95 | 22.06 | 20.89 | 20.56 | 5.58 |
| allreduce | exe err | | | -- | -- |
| amgmk | 4.10 | 0.83 | 0.77 | 4.32 | 1.10 |
| ans | 6.86 | 3.28 | 3.30 | exe err | 3.47 |
| aobench | 0.43 | 0.48 | 0.49 | 7.29 | 31.38 |
| aop | 8.82 | 34.75 | 35.36 | | |
| asmooth | 14.77 | 12.59 | 13.63 | 8.29 | 17.48 |
| assert | 4.18 | exe err | exe err | -- | -- |
| asta | 5.88 | 3.70 | 3.67 | 12.85 | 2.84 |
| atan2 | 4.31 | 0.78 | 0.76 | 4.25 | 0.77 |
| atomicAggregate | 7.57 | 5.45 | build err | -- | -- |
| atomicCAS | 17.43 | 14.78 | 14.05 | -- | -- |
| atomicCost | 6.96 | 8.25 | 8.58 | 9.37 | 198.36 |
| atomicIntrinsics | 89.15 | 70.23 | 70.23 | 4.28 | 0.19 |
| atomicPerf | 11.03 | 5.26 | over 600 | 11.82 | 40.75 |
| atomicReduction | 4.13 | 0.92 | 0.92 | 4.29 | 1.34 |
| atomicSystemWide | 5.24 | 5.25 | 10.09 | -- | -- |
| attention | 4.17 | 19.23 | 19.47 | 0.00 | 0.38 |
| attentionMultiHead | 5.26 | 1.69 | build err | -- | -- |
| axhelm | 3.94 | 0.80 | 0.81 | 4.05 | 0.84 |
| b+tree | 4.72 | 0.97 | 0.96 | 4.12 | 0.88 |
| babelstream | 4.54 | 0.95 | 0.95 | 4.59 | 17.43 |
| background-subtract | 5.40 | 4.73 | 4.74 | 7.66 | 4.63 |
| backprop | 4.07 | 0.47 | 0.46 | 4.63 | 0.40 |
| bezier-surface | 12.61 | 0.18 | 0.19 | 0.00 | 0.18 |
| bfs | 4.10 | 1.18 | 1.16 | 4.26 | 1.08 |
| bh | 7.96 | over 600 | over 600 | -- | -- |
| bicgstab | 4.75 | build err | over 600 | -- | -- |
| bilateral | 18.73 | 46.69 | 65.93 | 17.81 | over 600 |
| bincount | 5.02 | 2.01 | 2.02 | -- | -- |
| binomial | 4.63 | 1.79 | 1.78 | -- | 1.33 |
| bitcracker | 17.62 | 24.08 | 24.10 | -- | -- |
| bitonic-sort | 15.02 | 13.41 | 13.40 | 10.20 | 42.52 |
| bitpacking | 6.31 | 4.13 | 4.08 | -- | -- |
| bitpermute | 19.89 | 21.10 | build err | -- | -- |
| black-scholes | 6.55 | 3.97 | 4.03 | 6.57 | 252.90 |
| blas-dot | 10.70 | 10.28 | build err | -- | -- |
| blas-fp8gemm | 4.21 | 5.43 | build err | -- | -- |
| blas-gemm | 7.36 | 4.59 | 4.52 | -- | -- |
| blas-gemmBatched | 15.29 | 21.60 | 21.49 | -- | -- |
| blas-gemmEx | 11.62 | over 600 | build err | -- | -- |
| blas-gemmEx2 | 11.68 | over 600 | build err | -- | -- |
| blas-gemmStridedBatched | 15.70 | 21.70 | build err | -- | -- |
| blockAccess | 4.19 | 8.16 | 8.17 | -- | -- |
| blockexchange | 37.54 | 0.67 | 0.67 | -- | -- |
| bm3d | exe err | exe err | exe err | -- | -- |
| bmf | exe err | exe err | build err | -- | -- |
| bn | 5.67 | exe err | exe err | exe err | over 600 |
| bonds | 10.37 | 20.41 | 19.30 | 15.29 | 373.83 |
| boxfilter | 4.12 | 0.46 | 0.45 | 4.93 | exe err |
| bscan | 9.48 | 1.80 | 0.45 | -- | -- |
| bsearch | over 600 | 4.62 | 4.61 | exe err | over 600 |
| bspline-vgh | 7.09 | 10.15 | 10.05 | 4.98 | 9.93 |
| bsw | 3.99 | 0.56 | build err | -- | -- |
| btree | 7.22 | -- | build err | -- | -- |
| burger | 98.96 | 21.58 | 22.11 | 76.05 | 67.00 |
| bwt | over 600 | 13.44 | 13.28 | over 600 | 41.51 |
| car | 10.39 | 20.05 | 21.58 | 9.95 | over 600 |
| cbsfil | 33.05 | 5.85 | 5.94 | 37.40 | 198.98 |
| cc | exe err | exe err | build err | -- | -- |
| ccl | exe err | | build err | -- | -- |
| ccs | 5.39 | over 600 | over 600 | 5.37 | over 600 |
| ccsd-trpdrv | 16.66 | 18.89 | 19.10 | 15.05 | 0.34 |
| ced | 4.92 | 2.83 | 2.83 | -- | -- |
| cfd | 4.42 | 1.49 | 1.51 | 4.87 | 2.66 |
| chacha20 | 6.53 | 8.24 | 8.15 | 8.74 | 0.38 |
| channelShuffle | 546.16 | over 600 | over 600 | 544.76 | over 600 |
| channelSum | 67.59 | 52.04 | 52.06 | 68.99 | 334.81 |
| che | 5.17 | 2.70 | 2.71 | 4.93 | 426.17 |
| chemv | 4.73 | 1.90 | 1.91 | 5.29 | 0.21 |
| chi2 | 202.09 | 20.48 | 20.44 | 254.41 | over 600 |
| clenergy | 4.33 | 1.34 | 1.34 | 5.04 | over 600 |
| clink | 6.83 | 5.03 | 5.04 | 8.09 | 6.11 |
| clock | 4.01 | 0.44 | 0.44 | -- | -- |
| cm | 0.00 | 0.18 | 0.18 | exe err | build err |
| cmembench | 6.34 | 10.24 | 10.21 | -- | -- |
| cmp | | | | | |
| cobahh | 489.22 | over 600 | over 600 | 421.00 | over 600 |
| collision | 10.03 | 2.60 | build err | -- | -- |
| colorwheel | 39.90 | 2.34 | 2.34 | 28.98 | 79.00 |
| columnarSolver | 11.52 | exe err | exe err | 11.60 | exe err |
| complex | 24.73 | 3.72 | 3.66 | 14.80 | over 600 |
| compute-score | 4.90 | 4.72 | 4.74 | 6.53 | 3.93 |
| concat | 18.53 | 21.03 | 21.07 | 20.23 | over 600 |
| concurrentKernels | 283.21 | 32.58 | 32.77 | -- | -- |
| contract | over 600 | 15.07 | 15.12 | over 600 | over 600 |
| conversion | 2.84 | 1.05 | 1.46 | 5.42 | 1.33 |
| convolution1D | over 600 | 101.11 | 101.07 | over 600 | build err |
| convolution3D | exe err | 75.19 | build err | 0.35 | over 600 |
| convolutionDeformable | build err | build err | build err | -- | -- |
| convolutionSeparable | 13.20 | 25.92 | 26.02 | 22.36 | 25.24 |
| cooling | over 600 | 1.28 | 1.88 | over 600 | 408.04 |
| coordinates | 480.88 | 3.34 | 3.34 | -- | -- |
| copy | 12.53 | 15.72 | over 600 | -- | -- |
| crc64 | 153.21 | 4.06 | 4.08 | 157.64 | 4.55 |
| cross | 14.19 | 11.29 | 11.06 | over 600 | 18.96 |
| crossEntropy | 8.04 | 10.40 | 9.76 | -- | -- |
| crs | 26.73 | 10.35 | 10.37 | over 600 | 11.55 |
| d2q9-bgk | 0.80 | 1.46 | 1.48 | 6.17 | 4.51 |
| d3q19-bgk | 8.46 | 24.92 | 24.98 | -- | -- |
| damage | 21.46 | 1.02 | 1.02 | 217.98 | 0.91 |
| daphne | build err | build err | build err | -- | -- |
| dct8x8 | 117.26 | 1.52 | 1.51 | exe err | 1.33 |
| ddbp | 15.93 | 4.87 | 4.85 | 16.18 | 282.63 |
| debayer | 4.13 | 0.93 | 1.07 | exe err | 1.29 |
| degrid | 12.59 | 49.40 | 51.88 | 19.90 | 119.24 |
| dense-embedding | 81.41 | 180.53 | 180.46 | over 600 | over 600 |
| depixel | over 600 | 37.52 | 37.56 | over 600 | 14.91 |
| deredundancy | 14.12 | 59.16 | 59.13 | 17.62 | 59.07 |
| determinant | 4.42 | 0.63 | 0.64 | -- | -- |
| diamond | exe err | exe err | exe err | exe err | build err |
| dispatch | 6.31 | 2.02 | 1.99 | -- | -- |
| distort | 4.20 | 0.52 | 0.52 | 4.33 | 21.89 |
| divergence | 6.53 | 1.55 | 1.63 | 4.85 | 0.22 |
| doh | 14.90 | 19.81 | 19.80 | 4.10 | 20.11 |
| dp | 55.27 | build err | build err | 31.68 | 213.50 |
| dpid | 4.88 | 1.02 | build err | -- | -- |
| dropout | 29.95 | 0.93 | 0.93 | -- | -- |
| dslash | 4.35 | 2.25 | 4.42 | 10.22 | 10.12 |
| dwconv | 34.93 | 23.11 | 23.20 | -- | -- |
| dwconv1d | build err | build err | build err | -- | -- |
| dxtc1 | 4.14 | 0.63 | 0.65 | -- | -- |
| dxtc2 | 4.20 | 0.49 | 0.49 | 4.33 | 0.59 |
| easyWave | 4.90 | 2.48 | 2.45 | 4.94 | 4.96 |
| ecdh | 16.37 | 2.97 | 2.95 | 19.94 | over 600 |
| egs | 4.87 | exe err | build err | -- | -- |
| eigenvalue | 23.73 | 13.43 | 13.43 | 23.32 | over 600 |
| eikonal | 150.00 | 10.00 | 10.28 | -- | -- |
| entropy | 49.93 | 10.33 | 10.37 | 48.94 | over 600 |
| epistasis | 57.42 | 190.50 | 191.26 | 126.73 | over 600 |
| ert | 7.11 | 3.46 | 3.47 | -- | -- |
| expdist | 9.24 | 7.25 | 7.24 | 15.19 | 4.89 |
| extend2 | 4.31 | | | | |
| extrema | 8.81 | 6.28 | 6.21 | 8.99 | 411.13 |
| f16max | 23.64 | 35.49 | 35.53 | -- | -- |
| f16sp | 32.74 | 30.62 | build err | -- | -- |
| face | 4.42 | 1.04 | 1.05 | exe err | exe err |
| fdtd3d | 30.53 | 5.01 | 4.83 | 46.00 | 4.81 |
| feynman-kac | 83.40 | 114.59 | 113.85 | 23.89 | over 600 |
| fft | 4.87 | 2.18 | 2.18 | 5.62 | 0.60 |
| fhd | over 600 | 1.11 | 1.09 | | over 600 |
| filter | exe err | 14.79 | 14.80 | exe err | 7.41 |
| flame | exe err | 3.97 | 3.96 | -- | -- |
| flip | exe err | 25.60 | 26.16 | exe err | over 600 |
| floydwarshall | exe err | 1.89 | 1.90 | exe err | 181.61 |
| floydwarshall2 | exe err | 6.22 | 6.40 | -- | -- |
| fluidSim | exe err | 25.68 | 25.65 | exe err | 44.27 |
| fpc | exe err | 1.40 | 1.40 | exe err | 0.69 |
| fpdc | exe err | 2.95 | 2.90 | exe err | exe err |
| frechet | exe err | 1.13 | 1.11 | exe err | 0.20 |
| fresnel | exe err | 6.66 | build err | exe err | 556.72 |
| frna | exe err | 345.96 | 347.68 | exe err | 398.22 |
| fsm | exe err | 6.99 | 7.00 | exe err | 0.32 |
| fwt | exe err | 1.61 | 1.61 | exe err | 13.32 |
| ga | exe err | 5.03 | 5.02 | exe err | 7.75 |
| gabor | exe err | 4.94 | 8.04 | exe err | over 600 |
| gamma-correction | exe err | 19.27 | 19.29 | exe err | 23.63 |
| gaussian | exe err | 5.86 | 5.71 | exe err | 44.27 |
| gc | exe err | exe err | build err | exe err | over 600 |
| gd | exe err | 14.40 | 30.33 | exe err | 29.85 |
| ge-spmm | exe err | 8.70 | 8.58 | -- | -- |
| geam | exe err | 20.72 | 20.68 | -- | -- |
| gels | exe err | 1.83 | build err | -- | -- |
| gelu | exe err | 179.10 | 178.80 | -- | -- |
| gemv | exe err | 78.72 | build err | -- | -- |
| geodesic | exe err | 3.85 | 3.88 | exe err | 4.06 |
| gerbil | exe err | exe err | build err | -- | -- |
| gibbs | exe err | 0.72 | 0.70 | -- | -- |
| glu | exe err | 47.56 | 118.56 | exe err | over 600 |
| gmm | exe err | 1.92 | 1.03 | exe err | 1.91 |
| goulash | exe err | 12.54 | 12.55 | exe err | over 600 |
| gpp | exe err | 239.61 | 237.66 | exe err | over 600 |
| graphB+ | exe err | over 600 | build err | -- | -- |
| graphExecution | exe err | 1.84 | 1.84 | -- | -- |
| grep | exe err | 24.47 | 24.22 | exe err | 5.06 |
| grrt | exe err | 55.78 | 52.08 | exe err | build err |
| gru | exe err | 48.58 | 48.62 | -- | -- |
| haccmk | 4.58 | 4.28 | 4.31 | 5.83 | 5.45 |
| halo-finder | 5.06 | 7.51 | 5.24 | -- | -- |
| hausdorff | 32.48 | 14.46 | 14.48 | 33.25 | 15.76 |
| haversine | 4.25 | 1.84 | 2.08 | 4.36 | 2.12 |
| hbc | 15.35 | 15.25 | 15.17 | -- | -- |
| heartwall | 5.53 | 0.51 | 0.52 | exe err | exe err |
| heat | 20.72 | 40.38 | 42.03 | 18.88 | 70.02 |
| heat2d | build err | 11.13 | 11.12 | 244.40 | 331.87 |
| hellinger | 6.73 | 6.63 | 6.68 | 6.33 | 6.51 |
| henry | 5.41 | 3.54 | 3.91 | 4.73 | 3.96 |
| hexciton | 5.65 | 7.43 | build err | 6.51 | 17.89 |
| histogram | 10.64 | 1.33 | 1.34 | 4.75 | 8.74 |
| hmm | 5.85 | 3.98 | 4.00 | 16.22 | 7.03 |
| hogbom | 4.28 | 0.95 | 0.93 | 4.51 | 1.06 |
| hotspot | 4.22 | 0.60 | 0.62 | -- | -- |
| hotspot3D | 21.97 | 23.36 | 23.23 | 18.68 | 27.38 |
| hpl | 75.13 | 112.16 | build err | -- | -- |
| hungarian | 4.09 | 0.71 | 0.80 | -- | -- |
| hwt1d | 5.26 | 3.51 | 3.41 | 5.12 | 3.62 |
| hybridsort | 12.89 | 16.60 | 16.61 | 12.91 | 16.75 |
| hypterm | 5.76 | 14.81 | 16.17 | 13.32 | 20.42 |
| idivide | 7.62 | 6.77 | 6.20 | 10.67 | exe err |
| interleave | 5.69 | 3.50 | 3.52 | 5.02 | 3.44 |
| interval | 9.28 | 47.21 | 47.77 | 12.24 | 54.57 |
| intrinsics-cast | 6.35 | 10.77 | 10.40 | -- | -- |
| intrinsics-simd | 29.35 | -- | build err | -- | -- |
| inversek2j | 4.14 | 1.05 | 0.65 | 11.58 | 25.97 |
| is | 4.01 | 1.41 | 1.42 | -- | -- |
| ising | 23.04 | 16.62 | 16.57 | 23.58 | 22.76 |
| iso2dfd | 19.31 | 51.79 | 54.28 | 19.13 | 57.27 |
| jaccard | 195.48 | 257.49 | build err | -- | -- |
| jacobi | 4.37 | 8.94 | build err | 5.18 | 6.18 |
| jenkins-hash | 6.37 | 5.10 | 5.09 | 6.60 | 5.62 |
| kalman | 15.78 | 27.95 | 27.99 | exe err | 10.00 |
| keccaktreehash | 7.79 | 10.78 | 10.73 | 7.86 | 10.84 |
| keogh | 209.33 | 85.80 | 86.17 | 43.32 | 87.61 |
| kernelLaunch | 15.52 | 23.24 | 19.11 | 75.30 | 115.25 |
| kmc | 5.49 | 6.61 | 6.39 | -- | -- |
| kmeans | 31.55 | 42.72 | 47.87 | 29.19 | 50.40 |
| knn | 4.55 | 11.77 | 11.01 | 4.80 | 11.48 |
| kurtosis | 17.99 | 169.03 | 169.18 | -- | -- |
| lanczos | 4.86 | 3.21 | 3.30 | 5.07 | 4.07 |
| langevin | 33.39 | 11.75 | 12.33 | 8.50 | 17.95 |
| langford | 4.44 | 2.53 | 2.57 | build err | exe err |
| laplace | 3.88 | 4.48 | 4.58 | 9.08 | 22.48 |
| laplace3d | 70.75 | 18.58 | 17.77 | 52.58 | 22.92 |
| lavaMD | 102.13 | 62.26 | 62.11 | 137.20 | 55.99 |
| layernorm | 5.02 | 4.52 | build err | -- | -- |
| layout | 4.69 | 1.42 | 1.43 | 4.56 | 1.65 |
| lci | 19.51 | 8.41 | 8.42 | 4.39 | 7.91 |
| lda | 4.85 | 2.06 | 2.09 | build err | 15.00 |
| ldpc | 4.91 | 10.53 | 11.26 | 10.90 | 17.01 |
| lebesgue | 7.72 | 16.92 | 18.45 | 6.12 | 18.56 |
| leukocyte | 4.48 | 1.03 | 1.04 | 4.80 | 1.14 |
| lfib4 | 19.62 | 30.87 | 30.92 | -- | -- |
| libor | 4.35 | 1.30 | 1.32 | 4.53 | 1.77 |
| lid-driven-cavity | 12.60 | 16.62 | 16.69 | 24.66 | 61.81 |
| lif | 89.87 | 76.77 | 76.60 | 127.27 | 81.06 |
| linearprobing | 104.36 | 75.05 | 76.27 | build err | exe err |
| log2 | | | | | |
| logan | | | | -- | -- |
| logic-resim | | | | -- | -- |
| logic-rewrite | | | | -- | -- |
| logprob | | | | -- | -- |
| lombscargle | 3.02 | 1.21 | 1.21 | 4.57 | 1.19 |
| loopback | 7.39 | 11.01 | 11.12 | 9.80 | 11.49 |
| lr | 8.05 | 6.97 | 6.76 | 32.49 | 49.56 |
| lrn | 101.58 | 50.28 | 50.35 | 76.43 | 63.30 |
| lsqt | 21.76 | 60.36 | build err | 57.04 | 75.07 |
| lud | 7.84 | 10.20 | 10.75 | 47.70 | 25.97 |
| ludb | 6.52 | 9.79 | build err | -- | -- |
| lulesh | 9.18 | 15.02 | 16.64 | 9.76 | 570.69 |
| lzss | 0.00 | 0.17 | 0.17 | -- | -- |
| mallocFree | 5.28 | 10.43 | build err | 4.18 | 0.73 |
| mandelbrot | 7.44 | 15.05 | 15.16 | 5.40 | 13.39 |
| marchingCubes | 11.94 | 7.83 | build err | -- | -- |
| mask | 159.29 | 91.00 | 93.05 | 141.62 | 94.77 |
| match | 42.31 | 70.46 | 70.48 | 42.46 | 80.75 |
| matern | 16.40 | 31.93 | 31.94 | 110.75 | 108.16 |
| matrix-rotate | 29.72 | 9.78 | 10.12 | 29.32 | 10.47 |
| matrixT | 31.64 | 22.14 | 22.29 | -- | -- |
| maxFlops | 27.14 | 42.59 | 42.41 | 26.94 | 43.45 |
| maxpool3d | 31.84 | 14.98 | 14.92 | 33.59 | 15.48 |
| mcmd | 14.03 | 109.42 | 109.92 | 15.80 | 111.90 |
| mcpr | 11.71 | 60.92 | 60.91 | 13.92 | 68.26 |
| md | 14.48 | 19.47 | 19.51 | 14.77 | 19.00 |
| md5hash | 17.10 | 33.94 | 33.97 | 17.04 | 42.12 |
| mdh | 82.17 | 219.20 | build err | 35.84 | 220.52 |
| meanshift | 5.33 | 3.20 | 3.42 | 6.24 | 3.02 |
| medianfilter | 5.55 | 14.20 | 14.34 | 6.40 | 4.31 |
| memcpy | 7.04 | 16.95 | 17.13 | 8.08 | 17.62 |
| memtest | 20.01 | 28.17 | 28.15 | 21.19 | 37.77 |
| merge | over 600 | over 600 | over 600 | over 600 | over 600 |
| merkle | 8.08 | 15.53 | 15.51 | -- | -- |
| metropolis | over 600 | 71.66 | build err | over 600 | 297.58 |
| mf-sgd | 8.27 | 0.34 | 0.32 | -- | -- |
| michalewicz | 43.75 | 130.75 | 130.48 | over 600 | 138.65 |
| miniDGS | build err | -- | build err | -- | -- |
| miniFE | | 9.94 | 9.90 | | 10.19 |
| miniWeather | exe err | 3.75 | | 0.01 | 127.97 |
| minibude | 4.69 | 2.46 | 2.46 | 5.24 | 3.04 |
| minimap2 | 4.36 | build err | 1.82 | 3.96 | 16.61 |
| minimod | 5.98 | 2.86 | 2.80 | -- | -- |
| minisweep | 291.01 | 91.83 | 92.98 | 186.49 | 19.54 |
| minkowski | 25.61 | 24.21 | 31.65 | 22.06 | over 600 |
| minmax | 214.67 | 230.68 | 230.40 | -- | -- |
| mis | 95.71 | 1.77 | 0.46 | exe err | |
| mixbench | 9.08 | 50.74 | 50.59 | 9.61 | 52.98 |
| mmcsf | 3.83 | 7.97 | 8.13 | -- | -- |
| mnist | 19.70 | 71.21 | 74.76 | -- | -- |
| morphology | 26.92 | 3.09 | 3.08 | over 600 | 0.80 |
| mpc | 2.83 | | exe err | -- | -- |
| mr | 4.73 | 1.31 | 1.27 | exe err | 1.62 |
| mrc | 81.61 | 4.15 | 4.13 | over 600 | 5.53 |
| mrg32k3a | 211.71 | 127.22 | 127.47 | -- | -- |
| mriQ | 6.35 | 6.84 | 27.88 | 6.97 | 31.07 |
| mt | 7.07 | 4.61 | 4.61 | 6.09 | 4.33 |
| mtf | 12.30 | 19.75 | 19.88 | -- | -- |
| multimaterial | 28.60 | 16.89 | 16.83 | 26.26 | 83.44 |
| multinomial | 4.16 | 1.73 | 1.73 | -- | -- |
| murmurhash3 | 4.71 | 2.91 | 2.83 | 4.60 | 23.00 |
| myocyte | 34.75 | 67.41 | 88.94 | 12.60 | 13.59 |
| nbnxm | 4.80 | 8.39 | over 600 | -- | -- |
| nbody | 46.53 | 0.51 | 0.51 | 60.77 | 9.98 |
| ne | 4.77 | 3.17 | 3.18 | 5.10 | 3.81 |
| nlll | 48.23 | 149.78 | 150.17 | over 600 | 147.80 |
| nms | 4.05 | 0.60 | 0.61 | 0.00 | 1.33 |
| nn | 0.00 | exe err | exe err | exe err | exe err |
| nonzero | 49.39 | 34.80 | 31.78 | -- | -- |
| norm2 | 5.47 | 3.40 | 3.40 | 6.51 | 5.17 |
| nosync | 9.34 | 5.47 | 7.41 | -- | -- |
| nqueen | 7.70 | 38.94 | 37.04 | 8.55 | over 600 |
| ntt | 8.02 | 7.71 | 7.71 | 8.86 | 16.98 |
| nw | 16.21 | 2.95 | 2.68 | 29.22 | 2.46 |
| openmp | 15.46 | 67.12 | 67.18 | 15.32 | 88.27 |
| opticalFlow | 4.11 | 3.09 | exe err | -- | -- |
| overlap | 4.23 | 0.57 | build err | -- | -- |
| overlay | 7.63 | 4.39 | 4.27 | 12.20 | 20.71 |
| p2p | 3.87 | 0.86 | 0.81 | -- | -- |
| p4 | 5.32 | 7.97 | 8.23 | over 600 | 2.56 |
| pad | 286.98 | 2.09 | build err | -- | -- |
| page-rank | 25.60 | 25.68 | 25.68 | 26.30 | 24.88 |
| particle-diffusion | 12.79 | 9.67 | 9.67 | 8.95 | 339.57 |
| particlefilter | 5.72 | 2.38 | 2.38 | exe err | 222.58 |
| particles | 1.32 | 2.27 | 1.96 | over 600 | exe err |
| pathfinder | 51.01 | 1.38 | 1.38 | exe err | 1.09 |
| pcc | 3.57 | 8.20 | 8.13 | -- | -- |
| perlin | 10.45 | 12.97 | 12.55 | -- | -- |
| permutate | 0.00 | 0.18 | 0.19 | 0.00 | 0.42 |
| permute | 35.81 | 1.20 | 1.06 | 42.31 | 1.80 |
| perplexity | 281.99 | 2.45 | 2.44 | 210.79 | 2.75 |
| phmm | 5.36 | 92.72 | 93.52 | 9.47 | 3.52 |
| pingpong | exe err | | build err | -- | -- |
| pitch | 5.39 | 4.11 | 4.11 | -- | -- |
| pnpoly | 8.58 | 20.88 | 20.94 | 119.16 | 226.22 |
| pns | 6.95 | 7.42 | 7.41 | exe err | 0.29 |
| pointwise | 19.08 | 1.97 | 2.88 | 192.38 | exe err |
| pool | 33.67 | 11.89 | 11.82 | 25.10 | over 600 |
| popcount | over 600 | 4.35 | 4.33 | 200.63 | over 600 |
| prefetch | 7.09 | 97.79 | 86.60 | -- | -- |
| present | 3.97 | 4.72 | 4.66 | 4.26 | 9.36 |
| prna | 81.06 | 512.06 | 519.13 | exe err | 555.41 |
| projectile | 0.79 | 1.09 | 1.09 | 4.17 | 134.77 |
| pso | 155.69 | 1.38 | 1.37 | 62.72 | 0.58 |
| qem | 9.46 | 11.72 | 11.36 | -- | -- |
| qkv | 23.69 | 35.83 | 35.35 | -- | -- |
| qrg | 8.74 | 19.99 | 19.98 | 11.90 | 26.81 |
| qtclustering | 37.30 | 0.93 | 0.95 | exe err | exe err |
| quicksort | 36.63 | 52.22 | 51.33 | exe err | 50.22 |
| radixsort | 0.86 | 1.67 | 1.68 | 9.85 | 4.99 |
| radixsort2 | 9.85 | 102.46 | 102.31 | -- | -- |
| rainflow | 42.82 | 7.21 | 7.24 | 35.46 | 66.42 |
| randomAccess | 10.26 | 13.65 | 13.68 | 26.20 | 19.77 |
| rayleighBenardConvection | 31.65 | 42.90 | 46.48 | -- | -- |
| reaction | 5.70 | 5.93 | 5.92 | 11.68 | 53.68 |
| recursiveGaussian | 5.48 | 4.51 | 4.50 | exe err | 4.41 |
| relu | over 600 | 18.79 | build err | -- | -- |
| remap | 49.03 | 21.88 | 21.07 | -- | -- |
| resize | 206.42 | 7.67 | 7.67 | exe err | 8.63 |
| resnet-kernels | 0.01 | 1.06 | 1.05 | -- | -- |
| reverse | 5.05 | 1.62 | 1.71 | 9.41 | 11.88 |
| reverse2D | 33.64 | 1.46 | 2.66 | -- | -- |
| rfs | 350.53 | 13.08 | 13.00 | exe err | 13.15 |
| ring | 18.31 | 5.71 | 5.73 | -- | -- |
| rle | 102.46 | 0.68 | build err | -- | -- |
| rng-wallace | 4.13 | 1.61 | 1.62 | 5.27 | 1.82 |
| rodrigues | 397.62 | 0.96 | 0.96 | 290.41 | 170.94 |
| romberg | 4.37 | 1.11 | 1.10 | 4.45 | 0.93 |
| rotary | 4.22 | 0.74 | 0.74 | -- | -- |
| rowwiseMoments | 219.38 | 2.42 | 2.43 | -- | -- |
| rsbench | 4.68 | 3.75 | 3.74 | exe err | 2.35 |
| rsc | 0.40 | 0.83 | 0.84 | 4.41 | 0.80 |
| rsmt | 5.18 | -- | build err | -- | -- |
| rtm8 | 62.35 | 70.92 | 70.86 | 52.14 | 132.89 |
| rushlarsen | over 600 | 11.10 | 11.15 | over 600 | 21.80 |
| s3d | 6.45 | 0.63 | 0.63 | 9.01 | 0.52 |
| s8n | over 600 | 28.55 | 28.48 | over 600 | 60.25 |
| sa | 52.50 | 2.57 | 2.54 | -- | -- |
| sad | 4.46 | 2.07 | 2.09 | over 600 | 2.02 |
| sampling | 7.59 | 7.82 | 7.73 | 7.91 | 2.55 |
| saxpy-ompt | exe err | 14.41 | 14.35 | -- | -- |
| sc | 1.10 | 1.13 | build err | -- | -- |
| scan | over 600 | 111.19 | 110.95 | over 600 | 112.23 |
| scan2 | 1.11 | 1.39 | 1.38 | 9.05 | 2.46 |
| scan3 | 4.82 | 1.29 | 1.28 | -- | -- |
| scel | 17.22 | 45.99 | 89.29 | over 600 | 46.78 |
| score | 12.34 | 4.76 | exe err | -- | -- |
| sddmm-batch | 203.98 | 214.90 | 211.63 | -- | -- |
| seam-carving | 4.12 | 0.49 | | -- | -- |
| secp256k1 | 5.38 | 1.27 | 1.29 | 6.39 | 0.21 |
| segment-reduce | 141.96 | 8.62 | 8.56 | -- | -- |
| segsort | 9.41 | 6.38 | build err | -- | -- |
| sheath | 7.35 | 4.92 | 490.81 | 7.63 | 330.97 |
| shmembench | 6.71 | 5.69 | 5.65 | | 7.75 |
| shuffle | exe err | 8.33 | build err | -- | -- |
| si | build err | 1.27 | build err | -- | -- |
| simpleMultiDevice | exe err | 6.59 | 6.58 | -- | -- |
| simpleSpmv | exe err | 13.40 | build err | exe err | 135.13 |
| simplemoc | exe err | 4.38 | 4.26 | exe err | 94.94 |
| slit | build err | build err | build err | -- | -- |
| slu | build err | 21.04 | 21.13 | build err | 27.47 |
| snake | 0.00 | 0.18 | 0.17 | 0.00 | 0.16 |
| sobel | 6.96 | 1.33 | 1.35 | 5.08 | 4.57 |
| sobol | 6.04 | 3.44 | 3.26 | 6.34 | 2.85 |
| softmax | 41.08 | 30.07 | build err | 41.12 | 117.58 |
| softmax-fused | 4.44 | 9.07 | build err | -- | -- |
| softmax-online | 28.85 | 20.62 | build err | -- | -- |
| sort | 6.24 | 6.04 | 6.04 | 38.45 | exe err |
| sortKV | 371.38 | 37.51 | 39.69 | -- | -- |
| sosfil | 6.15 | 6.58 | 6.57 | 15.19 | 10.22 |
| sparkler | 0.04 | 0.00 | 0.00 | -- | -- |
| spaxpby | over 600 | 133.14 | 131.40 | -- | -- |
| spd2s | over 600 | 105.01 | 119.79 | -- | -- |
| spgeam | 70.04 | 19.43 | 17.52 | -- | -- |
| spgemm | over 600 | 16.69 | 16.74 | -- | -- |
| sph | 3.07 | 3.06 | 3.06 | 7.16 | 3.71 |
| split | 50.62 | 1.22 | 1.22 | 195.54 | exe err |
| spm | 180.15 | 2.16 | 2.18 | 172.87 | 3.34 |
| spmm | 8.05 | 7.15 | 7.37 | -- | -- |
| spmv | over 600 | 6.19 | 6.17 | -- | -- |
| spnnz | 332.94 | 100.03 | 97.23 | -- | -- |
| sps2d | over 600 | 117.02 | 114.81 | -- | -- |
| spsm | over 600 | 116.96 | build err | -- | -- |
| spsort | 485.42 | 100.99 | 103.47 | -- | -- |
| sptrsv | 4.23 | 1.59 | 1.66 | 4.21 | 2.05 |
| srad | 61.26 | 0.55 | 0.55 | 0.00 | 0.00 |
| ss | 9.14 | 6.27 | 6.26 | 19.70 | 9.87 |
| ssim | 6.08 | 6.65 | build err | -- | -- |
| sss | 33.21 | 7.20 | 7.25 | -- | -- |
| sssp | 10.92 | 13.02 | 13.91 | -- | -- |
| stddev | 47.94 | 24.27 | 33.61 | 49.68 | 38.82 |
| stencil1d | 5.33 | 2.66 | 2.68 | 46.29 | 37.24 |
| stencil3d | 19.29 | 7.94 | 7.34 | exe err | 4.88 |
| streamCreateCopyDestroy | 1.88 | 11.59 | 11.66 | -- | -- |
| streamOrderedAllocation | 6.53 | 18.81 | 19.40 | -- | -- |
| streamPriority | 4.59 | 2.66 | 1.45 | -- | -- |
| streamUM | 42.58 | 42.65 | 92.93 | -- | -- |
| streamcluster | 20.49 | 26.83 | 27.00 | exe err | exe err |
| stsg | build err | build err | build err | -- | -- |
| su3 | 5.90 | 2.31 | 2.28 | 14.44 | 3.13 |
| surfel | over 600 | 10.30 | 10.29 | over 600 | over 600 |
| svd3x3 | 1.95 | 2.68 | 2.91 | 4.46 | 2.84 |
| sw4ck | 5.80 | 24.80 | 24.50 | exe err | over 600 |
| swish | 27.65 | 2.83 | 2.86 | over 600 | 2.32 |
| tensorAccessor | 440.46 | 17.35 | 17.31 | -- | -- |
| tensorT | 4.03 | 1.82 | 1.85 | 4.82 | 7.39 |
| testSNAP | 5.80 | 5.80 | 5.51 | 5.67 | 0.46 |
| thomas | over 600 | 17.64 | 17.60 | over 600 | 349.24 |
| threadfence | 45.02 | 0.87 | 0.86 | 36.34 | 0.51 |
| tissue | 13.34 | 20.94 | 20.60 | 15.51 | 20.20 |
| tonemapping | 4.74 | 10.75 | 10.73 | 4.76 | 11.46 |
| tpacf | 8.56 | 8.95 | build err | -- | -- |
| tqs | 4.16 | 1.65 | 1.39 | 0.00 | 3.08 |
| triad | 4.40 | 1.36 | 1.44 | 4.48 | 2.88 |
| tridiagonal | 90.52 | 27.32 | 27.72 | 170.65 | 35.94 |
| tsa | exe err | 1.80 | 1.80 | exe err | 1.80 |
| tsne | 1.36 | 0.58 | 0.57 | -- | -- |
| tsp | 0.00 | 0.19 | 0.21 | build err | 0.44 |
| unfold | 35.71 | 0.73 | 0.64 | -- | -- |
| urng | 4.25 | 0.47 | 0.46 | 4.27 | 0.58 |
| vanGenuchten | 27.95 | 5.63 | 5.63 | 43.07 | 6.06 |
| vmc | 4.35 | 1.82 | 1.84 | exe err | 1.74 |
| vol2col | 4.37 | 10.08 | 10.08 | 7.85 | exe err |
| vote | 5.52 | 12.01 | build err | -- | -- |
| voxelization | 0.00 | 0.18 | 0.18 | -- | -- |
| warpexchange | 310.74 | 0.73 | 0.69 | -- | -- |
| warpsort | 5.56 | 1.14 | build err | -- | -- |
| wedford | 194.37 | 15.99 | build err | -- | -- |
| winograd | 4.66 | 0.98 | 0.99 | 4.68 | 0.68 |
| wlcpow | 6.13 | 8.42 | 8.17 | 12.28 | 8.44 |
| wmma | 100.30 | 5.00 | build err | -- | -- |
| word2vec | 0.01 | 0.19 | 0.19 | -- | -- |
| wordcount | 20.92 | 9.16 | 9.08 | 21.51 | 13.78 |
| wsm5 | 7.06 | 9.66 | 9.67 | 8.77 | 10.66 |
| wyllie | over 600 | 3.52 | 3.51 | over 600 | 2.93 |
| xlqc | 3.11 | 0.51 | 0.51 | 3.14 | 0.49 |
| xsbench | 48.38 | 48.13 | 43.65 | 39.95 | 405.34 |
| zerocopy | 21.73 | 42.15 | over 600 | -- | -- |
| zeropoint | 18.78 | 151.85 | 152.15 | over 600 | 153.15 |
| zmddft | 4.46 | 3.80 | 3.80 | 15.22 | 3.00 |
| zoom | 41.09 | 8.16 | 8.11 | -- | -- |
