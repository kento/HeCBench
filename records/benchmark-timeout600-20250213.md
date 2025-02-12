| name | cuda | hip | hipified | omp_nvc | omp_aomp |
| -- | -- | -- | -- | -- | -- |
| accuracy | 2.51 | 6.52 | build err | 58.67 | 6.54 |
| ace | 3.12 | 3.32 | 3.31 | 3.18 | build err |
| adam | 5.14 | 4.21 | 4.20 | 6.73 | build err |
| addBiasResidualLayerNorm | 11.92 | 50.36 | build err | -- | -- |
| adv | 3.07 | 12.86 | 13.09 | 3.38 | 8.27 |
| aes | build err | build err | build err | build err | build err |
| affine | 3.49 | 4.75 | 4.75 | 12.58 | 51.58 |
| aidw | 4.39 | 9.71 | 48.18 | 4.50 | 11.25 |
| aligned-types | 3.66 | 2.15 | 2.17 | 4.61 | 3.11 |
| all-pairs-distance | 4.44 | 21.94 | 21.87 | 18.27 | 38.23 |
| allreduce | build err | exe err | exe err | -- | -- |
| amgmk | 2.51 | 0.81 | 0.75 | 2.43 | 0.80 |
| ans | 4.98 | over 600 | build err | exe err | |
| aobench | 0.22 | 0.49 | 0.48 | 5.31 | 0.48 |
| aop | 6.93 | 35.29 | 35.35 | | |
| asmooth | 12.65 | 12.76 | 13.82 | 8.23 | build err |
| assert | 2.48 | exe err | exe err | -- | -- |
| asta | 4.26 | 3.53 | 3.66 | 10.18 | 4.97 |
| atan2 | 2.56 | 0.76 | 0.76 | 2.55 | build err |
| atomicAggregate | 5.74 | 5.46 | build err | -- | -- |
| atomicCAS | 15.76 | 13.46 | 14.71 | -- | -- |
| atomicCost | 16.07 | 7.78 | 8.00 | 5.23 | 9.29 |
| atomicIntrinsics | 87.57 | 70.19 | 70.21 | 2.43 | 0.46 |
| atomicPerf | 9.20 | 5.26 | over 600 | 9.54 | over 600 |
| atomicReduction | 2.48 | 1.42 | 0.91 | 2.47 | 1.22 |
| atomicSystemWide | 3.45 | 10.92 | 10.42 | -- | -- |
| attention | build err | 158.73 | 162.49 | | build err |
| attentionMultiHead | 3.41 | 1.67 | build err | -- | -- |
| axhelm | build err | build err | build err | build err | build err |
| b+tree | build err | build err | build err | build err | build err |
| babelstream | 0.44 | 1.01 | 0.95 | 2.67 | 1.11 |
| background-subtract | 3.63 | 4.72 | 4.72 | 6.05 | build err |
| backprop | 2.44 | 0.46 | 0.46 | 2.40 | 0.45 |
| bezier-surface | exe err | exe err | exe err | exe err | build err |
| bfs | build err | build err | build err | build err | build err |
| bh | 3.96 | over 600 | build err | -- | -- |
| bicgstab | build err | build err | build err | -- | -- |
| bilateral | 14.72 | 52.29 | 65.92 | 16.11 | build err |
| bincount | 3.36 | 2.01 | 2.01 | -- | -- |
| binomial | 3.02 | 1.77 | 1.77 | -- | build err |
| bitcracker | 15.99 | 24.08 | 24.07 | -- | -- |
| bitonic-sort | 13.61 | 13.45 | 13.40 | 8.44 | build err |
| bitpacking | 4.23 | 4.08 | 4.08 | -- | -- |
| bitpermute | 18.31 | 21.27 | build err | -- | -- |
| black-scholes | 4.77 | 3.98 | 4.00 | 4.86 | 4.16 |
| blas-dot | 8.96 | 10.12 | build err | -- | -- |
| blas-fp8gemm | 2.48 | 5.38 | build err | -- | -- |
| blas-gemm | 5.55 | 4.53 | 4.54 | -- | -- |
| blas-gemmBatched | 13.63 | 21.78 | 21.57 | -- | -- |
| blas-gemmEx | 9.93 | over 600 | build err | -- | -- |
| blas-gemmEx2 | 9.90 | over 600 | build err | -- | -- |
| blas-gemmStridedBatched | 13.93 | 22.06 | build err | -- | -- |
| blockAccess | 2.47 | 8.16 | 8.18 | -- | -- |
| blockexchange | 5.59 | 0.67 | 0.67 | -- | -- |
| bm3d | build err | exe err | exe err | -- | -- |
| bmf | exe err | exe err | build err | -- | -- |
| bn | 2.06 | exe err | exe err | exe err | build err |
| bonds | 9.99 | 20.54 | 18.84 | 11.65 | 17.84 |
| boxfilter | 2.46 | 0.43 | 0.43 | 2.84 | 2.17 |
| bscan | 7.51 | 1.78 | 0.44 | -- | -- |
| bsearch | over 600 | 4.61 | 4.61 | exe err | 4.67 |
| bspline-vgh | 7.12 | 10.11 | 10.10 | 3.24 | 10.11 |
| bsw | exe err | exe err | build err | -- | -- |
| btree | 5.42 | -- | build err | -- | -- |
| burger | 97.94 | 21.47 | 21.48 | 73.75 | build err |
| bwt | over 600 | 13.76 | 13.22 | over 600 | 13.85 |
| car | 7.76 | 20.77 | 20.16 | 7.43 | build err |
| cbsfil | 35.78 | 5.92 | 5.97 | 33.69 | build err |
| cc | exe err | exe err | build err | -- | -- |
| ccl | build err | build err | build err | -- | -- |
| ccs | build err | build err | build err | build err | build err |
| ccsd-trpdrv | 18.94 | 18.95 | 18.94 | 14.54 | 17.94 |
| ced | build err | build err | build err | -- | -- |
| cfd | exe err | exe err | exe err | exe err | exe err |
| chacha20 | 2.66 | 8.33 | 8.23 | 6.80 | 13.42 |
| channelShuffle | 545.46 | over 600 | over 600 | 538.60 | over 600 |
| channelSum | 63.59 | 52.13 | 52.22 | 66.81 | build err |
| che | 3.83 | 2.69 | 2.73 | 3.42 | 3.30 |
| chemv | 3.11 | 1.90 | 1.89 | 3.21 | build err |
| chi2 | 199.68 | 20.47 | 20.46 | 254.57 | build err |
| clenergy | 2.67 | 1.33 | 1.33 | 3.21 | 2.90 |
| clink | exe err | exe err | exe err | exe err | exe err |
| clock | 0.20 | 0.49 | 0.44 | -- | -- |
| cm | exe err | exe err | exe err | build err | build err |
| cmembench | 2.29 | 10.21 | 10.18 | -- | -- |
| cmp | | | | | |
| cobahh | 485.08 | over 600 | over 600 | 421.18 | over 600 |
| collision | 8.77 | 2.80 | build err | -- | -- |
| colorwheel | 39.62 | 2.35 | 2.33 | 28.85 | 2.37 |
| columnarSolver | build err | build err | build err | build err | build err |
| complex | 20.28 | 3.67 | 3.65 | 12.94 | 2.62 |
| compute-score | 3.26 | 4.72 | 4.73 | 4.34 | build err |
| concat | 16.78 | 21.03 | 21.02 | 19.18 | 29.39 |
| concurrentKernels | 281.65 | 31.57 | 31.56 | -- | -- |
| contract | over 600 | 15.10 | 15.12 | over 600 | 18.15 |
| conversion | 5.02 | 0.95 | 0.93 | 3.35 | 0.68 |
| convolution1D | over 600 | 101.30 | 101.19 | over 600 | build err |
| convolution3D | build err | 75.11 | build err | 2.43 | build err |
| convolutionDeformable | build err | build err | build err | -- | -- |
| convolutionSeparable | 12.91 | 26.03 | 26.08 | 21.87 | build err |
| cooling | over 600 | 1.06 | 1.88 | over 600 | build err |
| coordinates | 477.72 | 3.34 | 3.35 | -- | -- |
| copy | 10.11 | 15.65 | over 600 | -- | -- |
| crc64 | 155.26 | 4.05 | 4.06 | 155.23 | 4.12 |
| cross | 7.44 | 11.25 | 11.35 | over 600 | build err |
| crossEntropy | 9.93 | 9.77 | 9.76 | -- | -- |
| crs | 24.95 | 10.36 | 10.36 | over 600 | 10.04 |
| d2q9-bgk | build err | build err | build err | build err | build err |
| d3q19-bgk | 4.59 | 24.96 | 25.01 | -- | -- |
| damage | 21.27 | 1.01 | 1.01 | 217.74 | 1.74 |
| daphne | | | build err | -- | -- |
| dct8x8 | 101.81 | 1.50 | 1.52 | build err | build err |
| ddbp | 15.81 | 4.88 | 4.72 | 14.97 | build err |
| debayer | 2.67 | 0.93 | 1.08 | build err | exe err |
| degrid | 12.55 | 49.95 | 51.89 | 17.77 | 89.41 |
| dense-embedding | 95.51 | 180.53 | 180.34 | over 600 | 193.95 |
| depixel | over 600 | 30.91 | 37.45 | over 600 | build err |
| deredundancy | build err | build err | build err | build err | build err |
| determinant | 0.35 | 0.72 | 0.65 | -- | -- |
| diamond | build err | build err | build err | build err | build err |
| dispatch | 2.43 | 1.98 | 1.98 | -- | -- |
| distort | 2.43 | 0.51 | 0.52 | 2.49 | 0.50 |
| divergence | 4.69 | 1.62 | 1.63 | 3.10 | 1.89 |
| doh | 14.74 | 19.83 | 19.81 | 2.44 | build err |
| dp | 53.01 | build err | build err | 29.58 | 8.08 |
| dpid | 3.14 | 1.02 | build err | -- | -- |
| dropout | 27.67 | 0.92 | 0.93 | -- | -- |
| dslash | 2.70 | 2.23 | 4.45 | 10.06 | 3.81 |
| dwconv | 32.10 | 23.11 | 23.09 | -- | -- |
| dwconv1d | | | | -- | -- |
| dxtc1 | exe err | exe err | exe err | -- | -- |
| dxtc2 | build err | build err | build err | exe err | exe err |
| easyWave | 1.84 | 2.75 | 2.46 | 3.04 | 2.69 |
| ecdh | 14.88 | 2.96 | 2.96 | 22.71 | 4.24 |
| egs | exe err | exe err | build err | -- | -- |
| eigenvalue | 19.64 | 13.26 | 13.42 | 21.79 | 16.76 |
| eikonal | 203.18 | 11.35 | 10.39 | -- | -- |
| entropy | 48.61 | 10.33 | 10.34 | 49.37 | build err |
| epistasis | 54.79 | 190.46 | 190.69 | 122.77 | build err |
| ert | 5.46 | 3.52 | 3.53 | -- | -- |
| expdist | 7.53 | 7.22 | 7.22 | 12.74 | build err |
| extend2 | | | | | |
| extrema | 6.83 | 6.21 | 6.39 | 7.09 | 7.23 |
| f16max | 22.02 | 35.51 | 35.53 | -- | -- |
| f16sp | 30.81 | 30.61 | build err | -- | -- |
| face | build err | build err | build err | build err | build err |
| fdtd3d | 29.88 | 5.17 | 4.82 | 45.08 | 4.73 |
| feynman-kac | 82.17 | 113.84 | 114.22 | 22.17 | build err |
| fft | 3.04 | 2.16 | 2.17 | 2.58 | 0.86 |
| fhd | over 600 | 1.09 | 1.09 | exe err | build err |
| filter | 171.75 | 14.73 | 14.84 | 155.12 | 11.26 |
| flame | 3.93 | 3.98 | 3.98 | -- | -- |
| flip | 45.10 | 26.34 | 25.67 | 45.24 | 36.14 |
| floydwarshall | 75.49 | 1.89 | 1.95 | 92.72 | 4.87 |
| floydwarshall2 | exe err | exe err | exe err | -- | -- |
| fluidSim | 18.83 | 26.84 | 25.70 | 17.60 | 26.47 |
| fpc | 65.86 | 1.41 | 1.40 | exe err | 1.47 |
| fpdc | 2.45 | 3.00 | 2.96 | build err | 13.95 |
| frechet | 8.91 | 1.11 | 1.11 | exe err | build err |
| fresnel | 4.58 | 6.65 | build err | build err | build err |
| frna | build err | build err | build err | build err | build err |
| fsm | 1.48 | 6.99 | 6.99 | build err | 11.12 |
| fwt | 3.10 | build err | 1.59 | 3.26 | build err |
| ga | 49.14 | 5.02 | 5.03 | 35.90 | 5.12 |
| gabor | 54.10 | 4.95 | 8.09 | build err | build err |
| gamma-correction | over 600 | 19.29 | 19.34 | over 600 | 17.84 |
| gaussian | 17.54 | 5.62 | 5.67 | 4.48 | build err |
| gc | exe err | exe err | build err | build err | exe err |
| gd | build err | build err | build err | build err | build err |
| ge-spmm | build err | build err | build err | -- | -- |
| geam | 64.83 | build err | build err | -- | -- |
| gels | 2.65 | 1.98 | build err | -- | -- |
| gelu | 117.70 | 178.96 | 178.94 | -- | -- |
| gemv | 7.92 | 78.53 | build err | -- | -- |
| geodesic | exe err | exe err | exe err | exe err | exe err |
| gerbil | build err | build err | build err | -- | -- |
| gibbs | build err | build err | build err | -- | -- |
| glu | 70.70 | 54.87 | 119.60 | over 600 | build err |
| gmm | build err | build err | build err | build err | build err |
| goulash | 29.94 | 15.06 | 12.41 | 36.83 | 20.44 |
| gpp | 4.82 | 236.98 | 239.31 | 4.16 | 4.23 |
| graphB+ | exe err | exe err | build err | -- | -- |
| graphExecution | 2.52 | 1.88 | 1.84 | -- | -- |
| grep | 19.53 | 24.59 | 24.58 | 1.20 | 11.08 |
| grrt | 14.43 | 55.83 | 51.95 | 12.84 | build err |
| gru | 21.19 | 48.63 | 48.58 | -- | -- |
| haccmk | 4.55 | 4.34 | 4.32 | 4.41 | 5.52 |
| halo-finder | 3.45 | 7.61 | 7.60 | -- | -- |
| hausdorff | 30.88 | 14.48 | 14.48 | 31.58 | 15.76 |
| haversine | 2.21 | 1.88 | 2.11 | 2.49 | 2.11 |
| hbc | 13.41 | 15.38 | 15.35 | -- | -- |
| heartwall | 2.45 | 0.62 | 0.59 | build err | exe err |
| heat | 20.80 | 40.17 | 39.91 | 17.22 | 69.24 |
| heat2d | build err | 11.17 | 11.10 | 243.75 | 330.76 |
| hellinger | 4.98 | 6.66 | 6.66 | 4.69 | 6.62 |
| henry | 3.56 | 3.56 | 4.55 | 2.60 | 4.59 |
| hexciton | 4.08 | 10.28 | build err | 4.81 | 17.98 |
| histogram | 9.25 | 1.33 | 1.32 | 3.22 | 8.70 |
| hmm | 4.16 | 3.93 | 3.98 | 14.60 | 6.98 |
| hogbom | 0.50 | 1.04 | 0.93 | 2.77 | 1.04 |
| hotspot | 2.55 | 0.60 | 0.62 | -- | -- |
| hotspot3D | 18.51 | 23.36 | 23.45 | 16.91 | 28.90 |
| hpl | 89.51 | 112.22 | build err | -- | -- |
| hungarian | 2.48 | 0.79 | 0.78 | -- | -- |
| hwt1d | 3.45 | 3.56 | 3.50 | 3.55 | 3.59 |
| hybridsort | 12.81 | 16.71 | 16.72 | 12.79 | 16.79 |
| hypterm | 5.69 | 16.18 | 16.17 | 13.85 | 20.88 |
| idivide | 5.98 | 6.75 | 6.74 | 9.22 | exe err |
| interleave | 4.10 | 3.52 | 3.49 | 3.33 | 3.42 |
| interval | 7.61 | 47.33 | 47.45 | 10.63 | 56.86 |
| intrinsics-cast | 4.82 | 9.32 | 10.31 | -- | -- |
| intrinsics-simd | 3.18 | -- | build err | -- | -- |
| inversek2j | 2.47 | 0.62 | 0.62 | 9.89 | 24.47 |
| is | 2.49 | 1.42 | 1.42 | -- | -- |
| ising | 22.72 | 15.79 | 16.65 | 23.83 | 22.46 |
| iso2dfd | 17.34 | 51.56 | 51.54 | 16.70 | 58.33 |
| jaccard | 195.75 | 258.14 | build err | -- | -- |
| jacobi | 2.76 | 8.94 | build err | 3.39 | 6.20 |
| jenkins-hash | 5.16 | 5.05 | 5.10 | 5.08 | 5.58 |
| kalman | 13.88 | 28.96 | 28.81 | exe err | 10.05 |
| keccaktreehash | 7.77 | 10.78 | 10.78 | 7.85 | 10.77 |
| keogh | 210.10 | 85.60 | 85.66 | 43.38 | 87.49 |
| kernelLaunch | 14.72 | 23.72 | 23.81 | 73.68 | 114.97 |
| kmc | 3.32 | 6.50 | 6.31 | -- | -- |
| kmeans | 28.48 | 47.80 | 47.18 | 27.86 | 49.98 |
| knn | 4.36 | 11.56 | 11.51 | 4.50 | 11.74 |
| kurtosis | 17.86 | 168.23 | 168.79 | -- | -- |
| lanczos | build err | build err | build err | build err | build err |
| langevin | 7.13 | 12.24 | 13.68 | 10.16 | 15.58 |
| langford | 2.84 | 2.58 | 2.60 | build err | exe err |
| laplace | 3.90 | 4.62 | 4.58 | 7.48 | 22.35 |
| laplace3d | 69.32 | 18.31 | 17.96 | 47.18 | 23.01 |
| lavaMD | 101.25 | 62.07 | 62.12 | 134.21 | 54.76 |
| layernorm | 3.30 | 4.11 | build err | -- | -- |
| layout | 3.08 | 1.43 | 1.42 | 2.88 | 1.65 |
| lci | 17.95 | 8.40 | 8.43 | 4.39 | exe err |
| lda | 3.08 | 2.11 | 2.09 | build err | 14.99 |
| ldpc | 4.79 | 6.66 | 6.63 | 9.42 | 26.49 |
| lebesgue | 5.48 | 16.89 | 16.93 | 4.39 | 16.96 |
| leukocyte | 0.62 | 1.20 | 1.13 | 3.14 | 1.26 |
| lfib4 | 17.85 | 31.59 | 31.57 | -- | -- |
| libor | 2.79 | 1.31 | 1.33 | 2.77 | 1.77 |
| lid-driven-cavity | 11.12 | 14.85 | 14.77 | 23.11 | 61.97 |
| lif | 97.64 | 76.63 | 76.66 | 110.84 | 81.68 |
| linearprobing | 104.15 | 77.35 | 77.96 | build err | exe err |
| log2 | 2.79 | 1.34 | 1.28 | exe err | 1.69 |
| logan | 6.70 | 15.57 | 16.31 | -- | -- |
| logic-resim | 7.03 | 7.77 | 7.55 | -- | -- |
| logic-rewrite | 47.86 | build err | build err | -- | -- |
| logprob | 13.56 | 16.68 | build err | -- | -- |
| lombscargle | 2.99 | 1.22 | 1.23 | 2.96 | 1.17 |
| loopback | 5.71 | 11.02 | 11.02 | 8.23 | 11.51 |
| lr | 6.16 | 6.95 | 6.96 | 30.71 | 48.80 |
| lrn | 101.30 | 50.11 | 50.17 | 76.87 | 63.55 |
| lsqt | 20.14 | 60.57 | build err | 55.22 | 70.86 |
| lud | 6.01 | 10.81 | 11.05 | 46.07 | 27.20 |
| ludb | 4.87 | 9.80 | build err | -- | -- |
| lulesh | 7.69 | 16.26 | 14.12 | 8.33 | exe err |
| lzss | build err | build err | build err | -- | -- |
| mallocFree | 1.23 | 8.27 | build err | 2.73 | 0.77 |
| mandelbrot | 5.71 | 15.46 | 15.10 | 3.73 | 13.40 |
| marchingCubes | 10.43 | 7.81 | build err | -- | -- |
| mask | 160.32 | 90.91 | 90.94 | 192.91 | 93.37 |
| match | 40.68 | 70.30 | 70.44 | 40.79 | 80.88 |
| matern | 14.68 | 31.93 | 31.93 | 109.19 | 108.13 |
| matrix-rotate | 30.42 | 9.92 | 9.76 | 30.11 | 10.29 |
| matrixT | 27.72 | 22.70 | 23.59 | -- | -- |
| maxFlops | 25.52 | 42.41 | 42.40 | 25.35 | 43.23 |
| maxpool3d | 32.57 | 14.93 | 14.86 | 32.07 | 15.39 |
| mcmd | 12.31 | 110.11 | 109.74 | 14.29 | 111.07 |
| mcpr | 10.13 | 60.91 | 60.87 | 12.25 | 68.27 |
| md | 14.53 | 19.56 | 19.54 | 14.84 | 18.98 |
| md5hash | 15.37 | 33.97 | 33.97 | 15.33 | 42.01 |
| mdh | 82.19 | 218.13 | build err | 35.64 | 219.75 |
| meanshift | 3.69 | 3.20 | 3.42 | 4.53 | 3.13 |
| medianfilter | 3.80 | 14.30 | 14.32 | 4.73 | 4.36 |
| memcpy | 5.44 | 18.67 | 19.03 | 6.49 | 17.64 |
| memtest | 18.37 | 28.12 | 28.11 | 19.47 | 37.70 |
| merge | over 600 | over 600 | over 600 | over 600 | over 600 |
| merkle | 8.17 | 15.52 | 15.55 | -- | -- |
| metropolis | over 600 | 71.55 | build err | over 600 | 356.11 |
| mf-sgd | | | | -- | -- |
| michalewicz | 44.22 | 130.55 | 130.63 | over 600 | 137.79 |
| miniDGS | | -- | | -- | -- |
| miniFE | 5.34 | build err | build err | build err | build err |
| miniWeather | build err | build err | build err | build err | 8.99 |
| minibude | 0.98 | 2.46 | 2.43 | 3.43 | 3.05 |
| minimap2 | exe err | build err | exe err | exe err | exe err |
| minimod | 1.98 | 2.80 | 2.88 | -- | -- |
| minisweep | 289.35 | 90.55 | 92.36 | 184.60 | 19.51 |
| minkowski | 24.47 | 24.22 | 31.68 | 20.89 | 25.04 |
| minmax | 219.19 | 231.16 | 231.11 | -- | -- |
| mis | exe err | exe err | exe err | exe err | exe err |
| mixbench | 5.26 | build err | 50.57 | 8.12 | 52.90 |
| mmcsf | 3.79 | 8.11 | 8.26 | -- | -- |
| mnist | build err | build err | build err | -- | -- |
| morphology | 23.04 | 3.14 | 3.11 | over 600 | 8.19 |
| mpc | build err | exe err | build err | -- | -- |
| mr | 3.07 | 1.27 | 1.28 | exe err | 1.25 |
| mrc | 82.00 | 4.16 | 4.15 | over 600 | 5.25 |
| mrg32k3a | 214.91 | 127.29 | 126.95 | -- | -- |
| mriQ | 4.63 | 6.83 | 27.83 | 5.31 | 31.02 |
| mt | 5.58 | 4.62 | 4.62 | 4.41 | 4.31 |
| mtf | 10.39 | 19.90 | 17.94 | -- | -- |
| multimaterial | 28.41 | 18.00 | 20.83 | 26.00 | 73.55 |
| multinomial | 2.62 | 1.75 | 1.75 | -- | -- |
| murmurhash3 | 2.73 | 2.92 | 2.95 | 2.73 | 3.55 |
| myocyte | 4.28 | 1.92 | 1.89 | | exe err |
| nbnxm | 3.07 | 8.49 | over 600 | -- | -- |
| nbody | 45.85 | 0.50 | 0.51 | 60.82 | 0.47 |
| ne | 4.77 | 3.18 | 3.64 | 4.72 | 3.86 |
| nlll | 48.72 | 146.03 | 149.92 | over 600 | 150.46 |
| nms | exe err | exe err | exe err | exe err | exe err |
| nn | exe err | exe err | exe err | exe err | exe err |
| nonzero | 46.12 | 36.95 | 30.36 | -- | -- |
| norm2 | 3.71 | 3.34 | 3.34 | 4.99 | 5.59 |
| nosync | 7.52 | 5.46 | 5.42 | -- | -- |
| nqueen | 6.31 | 37.65 | 36.99 | 6.85 | 46.84 |
| ntt | 6.07 | 7.72 | 7.71 | 6.62 | 25.54 |
| nw | 20.78 | 2.89 | 3.07 | 6.31 | 3.04 |
| openmp | 13.95 | 67.25 | 67.27 | 13.78 | 67.16 |
| opticalFlow | build err | build err | build err | -- | -- |
| overlap | 0.23 | 0.64 | build err | -- | -- |
| overlay | 5.80 | 4.43 | 4.40 | 10.75 | 20.72 |
| p2p | 2.27 | 0.80 | 0.81 | -- | -- |
| p4 | 3.70 | 7.96 | 8.25 | over 600 | 9.46 |
| pad | 288.41 | 2.00 | build err | -- | -- |
| page-rank | 26.65 | 25.71 | 25.72 | 25.86 | 25.04 |
| particle-diffusion | 11.47 | 9.60 | 9.71 | 9.27 | 9.87 |
| particlefilter | 3.60 | 2.38 | 2.38 | exe err | 2.80 |
| particles | 1.34 | 2.02 | 2.01 | over 600 | exe err |
| pathfinder | 50.90 | 1.34 | 1.36 | exe err | 1.67 |
| pcc | 3.68 | 8.18 | 8.24 | -- | -- |
| perlin | 8.65 | 13.04 | 13.01 | -- | -- |
| permutate | build err | build err | build err | build err | build err |
| permute | 35.71 | 1.16 | 1.04 | 40.85 | 0.98 |
| perplexity | 282.38 | 2.43 | 2.44 | 211.77 | 2.75 |
| phmm | 3.65 | 93.39 | 93.39 | 7.49 | 14.37 |
| pingpong | build err | build err | build err | -- | -- |
| pitch | 3.75 | 4.12 | 4.11 | -- | -- |
| pnpoly | 6.99 | 20.89 | 20.86 | 119.51 | 225.61 |
| pns | 5.36 | 7.42 | 7.42 | exe err | exe err |
| pointwise | 19.08 | 1.96 | 1.94 | 237.18 | exe err |
| pool | 27.59 | 12.18 | 11.79 | 37.41 | 13.78 |
| popcount | over 600 | 4.37 | 4.31 | 184.46 | 4.77 |
| prefetch | 5.46 | 206.20 | 224.77 | -- | -- |
| present | 2.87 | 4.68 | 4.67 | 3.05 | 4.76 |
| prna | build err | build err | build err | build err | build err |
| projectile | 0.83 | 1.13 | 1.12 | 2.54 | 1.17 |
| pso | 157.43 | 1.38 | 1.38 | 61.75 | 5.70 |
| qem | 7.77 | 11.92 | 11.67 | -- | -- |
| qkv | 22.34 | 35.54 | 35.46 | -- | -- |
| qrg | 7.26 | 19.98 | 20.04 | 10.37 | 26.77 |
| qtclustering | 35.63 | 0.95 | 0.95 | exe err | exe err |
| quicksort | 36.75 | 12.00 | 12.06 | exe err | build err |
| radixsort | 0.85 | 1.69 | 1.69 | 8.20 | 5.59 |
| radixsort2 | 24.73 | 102.86 | 102.31 | -- | -- |
| rainflow | 43.35 | 7.21 | 7.24 | 35.25 | 7.60 |
| randomAccess | 8.45 | 13.05 | 13.54 | 24.58 | 30.23 |
| rayleighBenardConvection | 29.90 | 46.53 | 46.45 | -- | -- |
| reaction | 4.12 | 5.94 | 5.93 | 9.19 | 10.85 |
| recursiveGaussian | exe err | exe err | exe err | exe err | exe err |
| relu | over 600 | 19.07 | build err | -- | -- |
| remap | 43.33 | 22.04 | 22.02 | -- | -- |
| resize | 209.52 | 7.74 | 7.72 | exe err | 8.64 |
| resnet-kernels | 0.01 | 1.04 | 1.05 | -- | -- |
| reverse | 3.49 | 1.71 | 1.71 | 7.96 | 10.45 |
| reverse2D | 32.35 | 1.47 | 2.66 | -- | -- |
| rfs | 353.32 | 12.82 | 13.80 | build err | 12.88 |
| ring | 18.35 | 5.74 | 5.73 | -- | -- |
| rle | 80.12 | 0.69 | build err | -- | -- |
| rng-wallace | 2.60 | 1.60 | 1.62 | 3.53 | 1.83 |
| rodrigues | 398.09 | 0.96 | 0.96 | 295.63 | 2.05 |
| romberg | 2.70 | 1.11 | 1.09 | 2.78 | 0.95 |
| rotary | 2.51 | 0.76 | 0.75 | -- | -- |
| rowwiseMoments | 204.87 | 2.44 | 2.43 | -- | -- |
| rsbench | 3.06 | 3.75 | 3.72 | build err | 2.36 |
| rsc | exe err | exe err | exe err | exe err | exe err |
| rsmt | exe err | -- | build err | -- | -- |
| rtm8 | 58.79 | 76.71 | 70.93 | 50.26 | 68.25 |
| rushlarsen | over 600 | 11.10 | 11.15 | over 600 | 11.08 |
| s3d | 2.65 | 0.62 | 0.60 | 2.85 | 0.76 |
| s8n | over 600 | 28.49 | 28.48 | over 600 | 30.30 |
| sa | build err | build err | build err | -- | -- |
| sad | | | | exe err | exe err |
| sampling | 5.79 | 7.97 | 7.95 | 7.90 | 10.43 |
| saxpy-ompt | exe err | 14.41 | 14.35 | -- | -- |
| sc | 0.92 | 1.13 | build err | -- | -- |
| scan | over 600 | 111.16 | 110.15 | over 600 | 182.23 |
| scan2 | 2.80 | 1.35 | 1.36 | 6.82 | 2.45 |
| scan3 | 2.86 | 1.24 | 1.28 | -- | -- |
| scel | 17.44 | 46.11 | 89.52 | over 600 | 46.84 |
| score | 10.66 | 4.73 | build err | -- | -- |
| sddmm-batch | 205.89 | 214.12 | 212.49 | -- | -- |
| seam-carving | | | | -- | -- |
| secp256k1 | 2.70 | 1.29 | 1.29 | 2.72 | 1.30 |
| segment-reduce | 158.61 | 8.79 | 8.64 | -- | -- |
| segsort | 6.51 | 6.41 | build err | -- | -- |
| sheath | 5.81 | 4.94 | 495.98 | 6.01 | 335.49 |
| shmembench | 5.14 | 5.66 | 5.65 | exe err | 7.72 |
| shuffle | 7.40 | 8.38 | build err | -- | -- |
| si | | | | -- | -- |
| simpleMultiDevice | 3.85 | 6.62 | 6.65 | -- | -- |
| simpleSpmv | over 600 | 13.06 | build err | over 600 | 11.46 |
| simplemoc | 214.79 | 4.33 | 4.36 | over 600 | 3.43 |
| slit | | build err | | -- | -- |
| slu | | build err | build err | | build err |
| snake | 0.00 | 0.17 | 0.18 | 0.00 | 0.40 |
| sobel | build err | build err | build err | build err | build err |
| sobol | 2.36 | 3.57 | 3.26 | 4.63 | 4.61 |
| softmax | 39.93 | 29.49 | build err | build err | build err |
| softmax-fused | 0.57 | 9.10 | build err | -- | -- |
| softmax-online | 26.12 | 21.21 | build err | -- | -- |
| sort | 4.65 | 6.07 | 6.07 | 36.84 | exe err |
| sortKV | 364.90 | 38.33 | 37.48 | -- | -- |
| sosfil | 4.42 | 6.64 | 6.64 | 13.58 | 10.27 |
| sparkler | build err | exe err | exe err | -- | -- |
| spaxpby | over 600 | 131.07 | 133.96 | -- | -- |
| spd2s | over 600 | 106.07 | 112.91 | -- | -- |
| spgeam | 69.36 | 17.64 | 17.86 | -- | -- |
| spgemm | over 600 | 16.59 | 16.70 | -- | -- |
| sph | 5.33 | 3.05 | 3.08 | 5.37 | 3.76 |
| split | 50.64 | 1.27 | 1.27 | 196.70 | 2.47 |
| spm | 176.39 | 2.19 | 2.19 | 172.52 | 3.39 |
| spmm | 6.57 | 7.31 | 7.20 | -- | -- |
| spmv | over 600 | 5.78 | 5.81 | -- | -- |
| spnnz | 331.98 | 105.98 | 98.74 | -- | -- |
| sps2d | over 600 | 108.74 | 109.97 | -- | -- |
| spsm | over 600 | 106.67 | build err | -- | -- |
| spsort | 489.63 | 102.81 | 104.08 | -- | -- |
| sptrsv | 0.00 | 0.18 | 0.17 | 0.00 | build err |
| srad | exe err | exe err | exe err | exe err | exe err |
| ss | exe err | exe err | exe err | exe err | exe err |
| ssim | 1.90 | 6.78 | build err | -- | -- |
| sss | build err | build err | build err | -- | -- |
| sssp | build err | build err | build err | -- | -- |
| stddev | 48.40 | 26.02 | 33.74 | 47.86 | 39.36 |
| stencil1d | 5.67 | 2.00 | 1.99 | build err | 88.74 |
| stencil3d | 17.70 | 6.87 | 7.70 | build err | 4.73 |
| streamCreateCopyDestroy | 1.87 | 17.36 | 17.22 | -- | -- |
| streamOrderedAllocation | 4.91 | 18.80 | 18.81 | -- | -- |
| streamPriority | 2.91 | 1.49 | 1.44 | -- | -- |
| streamUM | 41.10 | 25.25 | 40.41 | -- | -- |
| streamcluster | 18.95 | 4.01 | 4.01 | build err | build err |
| stsg | build err | build err | build err | -- | -- |
| su3 | 5.91 | 2.28 | 2.33 | 12.35 | 3.12 |
| surfel | over 600 | 10.44 | 10.43 | over 600 | 15.12 |
| svd3x3 | exe err | exe err | exe err | exe err | exe err |
| sw4ck | 2.37 | 26.47 | 25.01 | exe err | 52.52 |
| swish | 27.88 | 2.86 | 2.83 | over 600 | 2.39 |
| tensorAccessor | 461.61 | 17.52 | 17.54 | -- | -- |
| tensorT | 2.57 | 1.85 | 1.85 | 2.89 | 1.85 |
| testSNAP | build err | build err | build err | build err | build err |
| thomas | over 600 | 17.60 | 17.62 | over 600 | 17.64 |
| threadfence | 60.92 | 0.87 | 0.86 | 34.73 | 0.53 |
| tissue | 12.02 | 21.02 | 21.02 | 13.12 | 20.00 |
| tonemapping | build err | build err | build err | build err | build err |
| tpacf | 3.30 | 9.04 | build err | -- | -- |
| tqs | exe err | exe err | exe err | exe err | exe err |
| triad | 0.35 | 1.44 | 1.43 | 2.78 | 3.14 |
| tridiagonal | 74.50 | 27.59 | 27.33 | 157.66 | 38.11 |
| tsa | exe err | 1.80 | 1.81 | exe err | 1.83 |
| tsne | build err | build err | build err | -- | -- |
| tsp | exe err | exe err | exe err | build err | exe err |
| unfold | 35.34 | 0.70 | 0.60 | -- | -- |
| urng | exe err | exe err | exe err | exe err | exe err |
| vanGenuchten | 28.44 | 6.05 | 5.61 | 40.69 | 6.08 |
| vmc | 2.73 | 1.86 | 1.86 | build err | build err |
| vol2col | 4.29 | 10.01 | 10.07 | 6.21 | exe err |
| vote | 3.92 | 12.15 | build err | -- | -- |
| voxelization | build err | build err | build err | -- | -- |
| warpexchange | 307.56 | 0.68 | 0.66 | -- | -- |
| warpsort | 3.56 | 1.14 | build err | -- | -- |
| wedford | 196.96 | 15.45 | build err | -- | -- |
| winograd | 3.03 | 0.99 | 0.99 | 3.09 | 0.97 |
| wlcpow | 4.46 | 8.08 | 8.37 | 12.30 | 8.41 |
| wmma | 98.83 | 5.11 | build err | -- | -- |
| word2vec | build err | build err | build err | -- | -- |
| wordcount | 17.85 | 9.15 | 9.35 | 19.66 | 9.26 |
| wsm5 | 5.49 | 9.77 | 9.77 | 7.17 | 10.77 |
| wyllie | over 600 | 3.58 | 3.58 | over 600 | 3.62 |
| xlqc | build err | build err | build err | build err | build err |
| xsbench | 46.34 | 45.09 | 42.03 | 38.33 | 56.48 |
| zerocopy | 19.45 | 84.10 | over 600 | -- | -- |
| zeropoint | 18.89 | 151.87 | 151.94 | over 600 | 152.97 |
| zmddft | 2.58 | 3.82 | 3.79 | 14.37 | 3.96 |
| zoom | 39.41 | 8.18 | 8.18 | -- | -- |
