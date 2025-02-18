# used memory [GB]

| name | cuda | hip | hipified | omp_nvc | omp_aomp |
| -- | -- | -- | -- | -- | -- |
| accuracy | 0.9 | 0.6 | build err | 0.9 | 1.2 |
| ace | 4.0 | 3.9 | 3.9 | 4.0 | build err |
| adam | 0.6 | 0.3 | 0.3 | 0.6 | build err |
| addBiasResidualLayerNorm | 0.7 | 0.4 | build err | -- | -- |
| adv | 4.9 | 5.0 | 5.0 | 5.3 | 5.5 |
| aes | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| affine | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| aidw | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| aligned-types | 0.7 | 0.4 | 0.4 | 0.7 | 1.0 |
| all-pairs-distance | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| allreduce | build err | 0.3 | 0.3 | -- | -- |
| amgmk | 0.6 | 0.3 | 0.3 | 0.6 | 0.8 |
| ans | 0.6 | 0.3 | 0.3 | exe err | 0.9 |
| aobench | 0.6 | 0.3 | 0.3 | 0.6 | 0.8 |
| aop | 0.6 | 0.3 | 0.3 | 0.7 | 1.0 |
| asmooth | 1.6 | 1.4 | 1.4 | 1.6 | 1.9 |
| assert | 0.6 | exe err | exe err | -- | -- |
| asta | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| atan2 | 0.7 | 0.5 | 0.5 | 0.7 | 1.1 |
| atomicAggregate | 0.6 | 0.3 | build err | -- | -- |
| atomicCAS | 0.6 | 0.3 | 0.3 | -- | -- |
| atomicCost | 14.6 | 22.4 | 22.4 | 21.7 | 15.6 |
| atomicIntrinsics | 0.6 | 0.3 | 0.3 | 0.6 | 0.8 |
| atomicPerf | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| atomicReduction | 0.8 | 0.5 | 0.5 | 0.8 | 1.1 |
| atomicSystemWide | 0.6 | 0.3 | 0.3 | -- | -- |
| attention | 1.1 | 0.8 | 0.8 | 1.1 | 1.4 |
| attentionMultiHead | 0.6 | 0.3 | build err | -- | -- |
| axhelm | 0.7 | 0.5 | 0.5 | 0.7 | 1.1 |
| b+tree | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| babelstream | 1.3 | 1.1 | 1.1 | 1.7 | 1.7 |
| background-subtract | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| backprop | 0.6 | 0.3 | 0.3 | 0.6 | 0.6 |
| bezier-surface | 1.3 | 1.1 | 1.1 | 1.3 | 1.7 |
| bfs | 0.6 | 0.3 | 0.3 | 0.6 | 0.8 |
| bh | 0.6 | 0.3 | 0.3 | -- | -- |
| bicgstab | 0.7 | build err | 0.3 | -- | -- |
| bilateral | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| bincount | 0.6 | 0.3 | 0.3 | -- | -- |
| binomial | 0.6 | 0.3 | 0.3 | -- | 0.9 |
| bitcracker | 0.8 | 0.6 | 0.6 | -- | -- |
| bitonic-sort | 0.7 | 0.4 | 0.4 | 0.7 | 1.0 |
| bitpacking | 1.1 | 0.9 | 0.9 | -- | -- |
| bitpermute | 2.6 | 2.5 | build err | -- | -- |
| black-scholes | 2.5 | 2.3 | 2.3 | 2.5 | 2.9 |
| blas-dot | 4.7 | 4.6 | build err | -- | -- |
| blas-fp8gemm | 0.6 | 1.1 | build err | -- | -- |
| blas-gemm | 1.1 | 0.9 | 0.9 | -- | -- |
| blas-gemmBatched | 2.6 | 2.4 | 2.4 | -- | -- |
| blas-gemmEx | 4.1 | 0.3 | build err | -- | -- |
| blas-gemmEx2 | 4.0 | 1.0 | build err | -- | -- |
| blas-gemmStridedBatched | 2.6 | 2.4 | build err | -- | -- |
| blockAccess | 0.9 | 0.6 | 0.6 | -- | -- |
| blockexchange | 19.6 | 0.8 | 0.8 | -- | -- |
| bm3d | build err | exe err | exe err | -- | -- |
| bmf | exe err | 0.3 | build err | -- | -- |
| bn | 0.6 | exe err | exe err | exe err | exe err |
| bonds | 0.8 | 0.9 | 0.9 | 0.9 | 1.3 |
| boxfilter | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| bscan | 0.6 | 0.3 | 0.3 | -- | -- |
| bsearch | 54.0 | 0.5 | 0.5 | exe err | 1.1 |
| bspline-vgh | 2.6 | 2.4 | 2.4 | 2.6 | 3.0 |
| bsw | 0.6 | 0.3 | build err | -- | -- |
| btree | 5.4 | -- | build err | -- | -- |
| burger | 8.4 | 2.4 | 2.4 | 8.4 | 3.0 |
| bwt | 1.3 | 0.4 | 0.4 | exe err | 1.0 |
| car | 6.6 | 6.6 | 6.6 | 6.6 | 7.2 |
| cbsfil | 2.4 | 0.6 | 0.6 | 2.4 | 1.2 |
| cc | 1.1 | 0.9 | build err | -- | -- |
| ccl | build err | 0.3 | 0.3 | -- | -- |
| ccs | 0.6 | 0.3 | 0.3 | 0.6 | 0.8 |
| ccsd-trpdrv | 2.8 | 0.3 | 0.3 | 2.8 | 0.8 |
| ced | 0.6 | 0.3 | 0.3 | -- | -- |
| cfd | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| chacha20 | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| channelShuffle | 18.7 | 15.1 | 16.7 | 17.3 | 14.0 |
| channelSum | 6.8 | 13.4 | 6.9 | 8.9 | 7.5 |
| che | 1.1 | 0.8 | 0.8 | 1.1 | 1.4 |
| chemv | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| chi2 | 51.0 | 1.9 | 1.9 | 51.0 | 2.4 |
| clenergy | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| clink | 1.8 | 2.3 | 2.3 | 1.8 | 2.2 |
| clock | 0.6 | 0.3 | 0.3 | -- | -- |
| cm | 0.6 | 0.3 | 0.3 | build err | build err |
| cmembench | 0.6 | 0.3 | 0.3 | -- | -- |
| cmp | 0.0 | 0.0 | 0.0 | 0.0 | build err |
| cobahh | 55.9 | 0.0 | 0.0 | 55.9 | 0.3 |
| collision | 0.6 | 0.3 | build err | -- | -- |
| colorwheel | 2.3 | 0.5 | 0.5 | 2.4 | 1.1 |
| columnarSolver | 1.6 | exe err | exe err | 3.2 | exe err |
| complex | 0.7 | 0.3 | 0.3 | 0.7 | 0.9 |
| compute-score | 1.2 | 1.0 | 1.0 | 1.1 | 1.5 |
| concat | 3.6 | 3.5 | 3.5 | 6.5 | 4.1 |
| concurrentKernels | 49.8 | 0.5 | 0.5 | -- | -- |
| contract | 1.8 | 0.3 | 0.3 | 4.3 | 0.9 |
| conversion | 16.6 | 24.4 | 24.4 | 20.6 | 16.9 |
| convolution1D | 32.6 | 4.6 | 2.4 | 56.6 | build err |
| convolution3D | build err | 0.9 | build err | 0.6 | 0.9 |
| convolutionDeformable | build err | build err | build err | -- | -- |
| convolutionSeparable | 3.6 | 1.1 | 1.1 | 3.6 | 1.7 |
| cooling | 32.6 | 0.3 | 0.3 | 32.6 | 0.9 |
| coordinates | 55.5 | 0.6 | 0.6 | -- | -- |
| copy | 1.3 | 1.1 | 1.1 | -- | -- |
| crc64 | 4.2 | 0.3 | 0.4 | 7.3 | 0.9 |
| cross | 7.6 | 1.0 | 1.0 | 6.7 | 2.0 |
| crossEntropy | 5.1 | 1.8 | 1.8 | -- | -- |
| crs | 0.7 | 0.4 | 0.4 | 0.7 | 1.0 |
| d2q9-bgk | 0.9 | 0.6 | 0.6 | 0.9 | 1.2 |
| d3q19-bgk | 1.2 | 1.0 | 1.0 | -- | -- |
| damage | 8.7 | 0.5 | 0.5 | 8.7 | 1.1 |
| daphne | 0.6 | 0.3 | 0.3 | -- | -- |
| dct8x8 | 32.8 | 0.8 | 0.8 | exe err | 1.4 |
| ddbp | 3.9 | 3.8 | 3.8 | 3.9 | 4.5 |
| debayer | 2.0 | 0.6 | 0.6 | exe err | exe err |
| degrid | 1.7 | 1.5 | 1.5 | 1.7 | 2.1 |
| dense-embedding | 14.6 | 10.1 | 10.1 | 5.4 | 15.7 |
| depixel | 8.2 | 0.4 | 0.4 | 9.7 | 1.0 |
| deredundancy | 9.7 | 0.5 | 0.5 | 3.5 | 1.1 |
| determinant | 0.8 | 0.3 | 0.3 | -- | -- |
| diamond | 0.7 | 0.5 | 0.7 | 0.7 | build err |
| dispatch | 0.6 | 0.3 | 0.3 | -- | -- |
| distort | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| divergence | 0.6 | 0.3 | 0.3 | 0.6 | 0.8 |
| doh | 0.6 | 0.3 | 0.3 | 0.6 | 0.8 |
| dp | 49.5 | 2.5 | 2.5 | 73.8 | 5.2 |
| dpid | 34.2 | 0.5 | build err | -- | -- |
| dropout | 51.2 | 0.4 | 0.4 | -- | -- |
| dslash | 1.9 | 3.1 | 1.7 | 3.1 | 3.5 |
| dwconv | 3.2 | 2.4 | 2.4 | -- | -- |
| dwconv1d | build err | build err | build err | -- | -- |
| dxtc1 | 0.6 | 0.3 | 0.4 | -- | -- |
| dxtc2 | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| easyWave | 0.8 | 0.5 | 0.5 | 0.8 | 1.1 |
| ecdh | 4.4 | 0.7 | 0.7 | 13.3 | 1.7 |
| egs | 1.2 | exe err | build err | -- | -- |
| eigenvalue | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| eikonal | 50.3 | 3.8 | 3.8 | -- | -- |
| entropy | 2.5 | 0.6 | 0.6 | 2.5 | 1.2 |
| epistasis | 4.3 | 4.2 | 4.2 | 4.3 | 4.8 |
| ert | 0.6 | 0.3 | 0.3 | -- | -- |
| expdist | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| extend2 | exe err | exe err | exe err | exe err | build err |
| extrema | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| f16max | 3.6 | 3.5 | 3.5 | -- | -- |
| f16sp | 1.6 | 1.4 | build err | -- | -- |
| face | 0.6 | 0.3 | 0.3 | build err | 0.8 |
| fdtd3d | 1.0 | 0.4 | 0.4 | 1.0 | 0.8 |
| feynman-kac | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| fft | 0.8 | 0.6 | 0.6 | 0.8 | 1.2 |
| fhd | 2.5 | 0.3 | 0.3 | 0.0 | 0.9 |
| filter | 11.3 | 1.1 | 1.1 | 11.4 | 1.7 |
| flame | 0.7 | 0.4 | 0.4 | -- | -- |
| flip | 34.1 | 17.5 | 17.5 | 50.8 | 18.0 |
| floydwarshall | 0.7 | 0.3 | 0.3 | 0.7 | 0.8 |
| floydwarshall2 | 0.6 | 0.4 | 0.4 | -- | -- |
| fluidSim | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| fpc | 2.6 | 0.4 | 0.4 | exe err | 1.0 |
| fpdc | 2.1 | 1.4 | 1.4 | build err | 2.0 |
| frechet | 2.8 | 0.3 | 0.3 | exe err | exe err |
| fresnel | 1.8 | 1.6 | build err | build err | 2.2 |
| frna | 3.9 | 3.8 | 3.8 | exe err | 4.4 |
| fsm | 8.7 | 7.3 | 7.3 | build err | 7.9 |
| fwt | build err | 0.4 | 0.4 | 0.6 | 1.0 |
| ga | 0.6 | 0.3 | 0.3 | 0.6 | 0.8 |
| gabor | 24.3 | 1.3 | 1.3 | build err | 2.0 |
| gamma-correction | 0.0 | 0.4 | 0.4 | 0.0 | 1.0 |
| gaussian | 0.7 | 0.4 | 0.4 | 0.7 | 1.0 |
| gc | 0.6 | exe err | build err | build err | 0.9 |
| gd | 0.8 | 0.5 | 0.5 | 0.8 | 1.1 |
| ge-spmm | 1.1 | 0.8 | 0.8 | -- | -- |
| geam | 28.5 | 4.6 | 4.6 | -- | -- |
| gels | 0.6 | 0.4 | build err | -- | -- |
| gelu | 4.7 | 4.6 | 4.6 | -- | -- |
| gemv | 1.6 | 1.1 | build err | -- | -- |
| geodesic | 0.8 | 0.5 | 0.5 | 0.8 | 1.1 |
| gerbil | exe err | 55.3 | build err | -- | -- |
| gibbs | 0.6 | 55.3 | 0.3 | -- | -- |
| glu | 15.8 | 8.9 | 8.9 | 0.0 | 9.5 |
| gmm | 8.2 | 0.3 | 0.3 | 8.2 | 0.9 |
| goulash | 26.2 | 5.7 | 5.7 | 26.2 | 6.3 |
| gpp | 2.7 | 2.5 | 2.5 | 2.7 | 3.2 |
| graphB+ | 0.6 | 0.3 | build err | -- | -- |
| graphExecution | 1.1 | 1.0 | 1.0 | -- | -- |
| grep | 0.6 | 0.3 | 0.3 | 0.0 | 0.3 |
| grrt | 0.6 | 0.6 | 0.5 | 0.6 | build err |
| gru | 9.2 | 2.6 | 1.5 | -- | -- |
| haccmk | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| halo-finder | 0.6 | 0.3 | 0.3 | -- | -- |
| hausdorff | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| haversine | 1.0 | 0.7 | 0.8 | 1.0 | 1.3 |
| hbc | 0.6 | 0.3 | 0.3 | -- | -- |
| heartwall | 3.3 | 0.3 | 17.4 | exe err | exe err |
| heat | 16.9 | 17.5 | 26.1 | 16.9 | 18.1 |
| heat2d | build err | 0.6 | 0.6 | 0.8 | 1.1 |
| hellinger | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| henry | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| hexciton | 0.6 | 0.3 | build err | 0.6 | 0.9 |
| histogram | 0.6 | 0.3 | 0.3 | 0.6 | 1.0 |
| hmm | 0.7 | 0.4 | 0.4 | 0.7 | 1.0 |
| hogbom | 0.7 | 0.4 | 0.4 | 0.7 | 1.0 |
| hotspot | 0.6 | 0.3 | 0.3 | -- | -- |
| hotspot3D | 0.6 | 0.3 | 0.3 | 0.6 | 0.8 |
| hpl | 9.5 | 16.9 | build err | -- | -- |
| hungarian | 0.6 | 0.3 | 0.3 | -- | -- |
| hwt1d | 0.7 | 0.4 | 0.4 | 0.7 | 1.0 |
| hybridsort | 1.2 | 0.9 | 0.9 | 1.2 | 1.5 |
| hypterm | 3.5 | 3.3 | 3.3 | 3.5 | 3.9 |
| idivide | 0.6 | 0.3 | 0.3 | 0.6 | exe err |
| interleave | 0.6 | 0.3 | 0.3 | 0.6 | 0.8 |
| interval | 1.0 | 0.6 | 0.6 | 1.0 | 1.2 |
| intrinsics-cast | 43.3 | 45.1 | 45.1 | -- | -- |
| intrinsics-simd | 15.8 | -- | build err | -- | -- |
| inversek2j | 0.6 | 22.7 | 0.3 | 0.6 | 0.9 |
| is | 1.7 | 1.4 | 1.4 | -- | -- |
| ising | 8.1 | 8.1 | 8.1 | 8.1 | 8.8 |
| iso2dfd | 49.7 | 51.8 | 51.8 | 49.7 | 52.4 |
| jaccard | 2.9 | 2.7 | build err | -- | -- |
| jacobi | 0.6 | 0.3 | build err | 0.6 | 0.9 |
| jenkins-hash | 10.0 | 10.2 | 10.2 | 10.0 | 10.8 |
| kalman | 10.8 | 11.2 | 11.2 | exe err | 11.4 |
| keccaktreehash | 0.7 | 0.4 | 0.4 | 0.7 | 1.0 |
| keogh | 3.6 | 3.5 | 3.5 | 5.7 | 5.7 |
| kernelLaunch | 0.6 | 0.3 | 0.3 | 0.6 | 0.8 |
| kmc | 0.7 | 0.4 | 0.4 | -- | -- |
| kmeans | 0.7 | 0.4 | 0.4 | 0.7 | 1.0 |
| knn | 0.6 | 0.4 | 0.4 | 0.6 | 1.0 |
| kurtosis | 48.6 | 50.7 | 50.7 | -- | -- |
| lanczos | 0.6 | 50.7 | 0.4 | 0.6 | 1.0 |
| langevin | 31.1 | 32.3 | 32.3 | 31.1 | 32.9 |
| langford | 2.1 | 3.5 | 3.5 | build err | exe err |
| laplace | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| laplace3d | 13.7 | 13.2 | 13.2 | 8.8 | 9.4 |
| lavaMD | 49.9 | 52.1 | 52.1 | 49.9 | 52.7 |
| layernorm | 0.6 | 28.5 | build err | -- | -- |
| layout | 0.6 | 0.4 | 0.4 | 0.6 | 0.9 |
| lci | 0.6 | 0.3 | 0.3 | 0.0 | exe err |
| lda | 0.6 | 0.4 | 0.4 | build err | 1.0 |
| ldpc | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| lebesgue | 9.7 | 68.2 | 68.2 | 50.9 | 68.1 |
| leukocyte | 0.6 | 46.3 | 0.3 | 0.6 | 0.9 |
| lfib4 | 8.7 | 8.9 | 8.9 | -- | -- |
| libor | 4.0 | 1.8 | 1.8 | 4.0 | 2.4 |
| lid-driven-cavity | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| lif | 23.4 | 24.3 | 24.3 | 23.4 | 24.9 |
| linearprobing | 1.4 | 0.9 | 0.9 | build err | exe err |
| log2 | 1.3 | 1.1 | 1.1 | exe err | 1.7 |
| logan | 2.7 | 2.5 | 2.6 | -- | -- |
| logic-resim | 0.8 | 0.7 | 0.7 | -- | -- |
| logic-rewrite | 19.4 | build err | build err | -- | -- |
| logprob | 8.4 | 8.5 | build err | -- | -- |
| lombscargle | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| loopback | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| lr | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| lrn | 22.5 | 23.3 | 23.3 | 25.5 | 23.9 |
| lsqt | 1.7 | 1.7 | build err | 2.0 | 2.4 |
| lud | 4.7 | 4.6 | 4.6 | 4.7 | 5.2 |
| ludb | 0.7 | 0.4 | build err | -- | -- |
| lulesh | 54.2 | 56.6 | 56.6 | 54.2 | exe err |
| lzss | 8.6 | 48.8 | 41.6 | -- | -- |
| mallocFree | 8.4 | 34.3 | build err | 16.6 | 7.4 |
| mandelbrot | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| marchingCubes | 0.6 | 0.3 | build err | -- | -- |
| mask | 16.9 | 17.4 | 17.4 | 25.6 | 17.9 |
| match | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| matern | 1.0 | 0.7 | 0.7 | 1.0 | 1.3 |
| matrix-rotate | 8.6 | 8.7 | 8.7 | 8.6 | 9.3 |
| matrixT | 15.9 | 16.3 | 16.3 | -- | -- |
| maxFlops | 0.6 | 0.3 | 0.3 | 0.6 | 0.8 |
| maxpool3d | 10.2 | 10.4 | 10.4 | 10.2 | 11.0 |
| mcmd | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| mcpr | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| md | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| md5hash | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| mdh | 0.6 | 0.3 | build err | 0.6 | 0.9 |
| meanshift | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| medianfilter | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| memcpy | 0.6 | 0.3 | 0.3 | 0.6 | 0.3 |
| memtest | 2.6 | 2.4 | 2.4 | 2.6 | 3.0 |
| merge | 15.8 | 16.3 | 16.3 | 15.8 | 16.9 |
| merkle | 1.6 | 1.6 | 1.6 | -- | -- |
| metropolis | 8.6 | 0.3 | build err | 8.6 | 0.9 |
| mf-sgd | 0.0 | 0.0 | 0.0 | -- | -- |
| michalewicz | 50.2 | 4.3 | 4.3 | 0.0 | 4.9 |
| miniDGS | build err | -- | build err | -- | -- |
| miniFE | 1.3 | 1.1 | 1.1 | 1.3 | 1.7 |
| miniWeather | build err | 0.3 | 0.3 | 0.6 | 0.3 |
| minibude | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| minimap2 | 1.0 | build err | 0.7 | 1.0 | 1.6 |
| minimod | 1.4 | 0.7 | 0.7 | -- | -- |
| minisweep | 10.8 | 4.6 | 4.6 | 10.8 | 5.2 |
| minkowski | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| minmax | 51.0 | 54.1 | 54.1 | -- | -- |
| mis | 0.6 | 53.3 | 0.3 | exe err | 0.6 |
| mixbench | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| mmcsf | 0.8 | 0.5 | 0.5 | -- | -- |
| mnist | 0.6 | 0.3 | 0.3 | -- | -- |
| morphology | 4.4 | 0.4 | 0.4 | 4.4 | 1.0 |
| mpc | 1.1 | 0.9 | exe err | -- | -- |
| mr | 0.6 | 0.3 | 0.3 | exe err | 0.9 |
| mrc | 38.6 | 0.5 | 0.5 | 0.0 | 1.1 |
| mrg32k3a | 6.0 | 11.6 | 11.6 | -- | -- |
| mriQ | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| mt | 0.7 | 0.4 | 0.4 | 0.7 | 1.0 |
| mtf | 0.6 | 0.3 | 0.3 | -- | -- |
| multimaterial | 3.1 | 2.9 | 2.9 | 3.1 | 3.5 |
| multinomial | 1.1 | 0.8 | 0.8 | -- | -- |
| murmurhash3 | 2.6 | 2.4 | 2.4 | 2.6 | 3.0 |
| myocyte | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| nbnxm | 0.6 | 0.3 | 0.3 | -- | -- |
| nbody | 0.6 | 0.3 | 0.3 | 0.6 | 0.8 |
| ne | 3.2 | 3.1 | 3.1 | 3.6 | 4.1 |
| nlll | 48.6 | 17.1 | 17.1 | 0.0 | 26.1 |
| nms | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| nn | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| nonzero | 0.7 | 0.4 | 0.4 | -- | -- |
| norm2 | 2.7 | 2.5 | 2.5 | 4.7 | 3.0 |
| nosync | 15.8 | 67.8 | 67.9 | -- | -- |
| nqueen | 0.6 | 59.2 | 0.4 | 0.6 | 1.0 |
| ntt | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| nw | 14.7 | 2.4 | 2.4 | 14.7 | 3.0 |
| openmp | 0.7 | 0.4 | 0.4 | 1.3 | 1.1 |
| opticalFlow | 0.6 | 0.3 | exe err | -- | -- |
| overlap | 0.7 | 0.4 | build err | -- | -- |
| overlay | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| p2p | 0.0 | 0.4 | 0.4 | -- | -- |
| p4 | 0.6 | 0.3 | 0.3 | 0.6 | 1.0 |
| pad | 3.3 | 0.3 | build err | -- | -- |
| page-rank | 12.8 | 13.1 | 13.1 | 12.8 | 13.7 |
| particle-diffusion | 3.3 | 3.2 | 3.2 | 3.3 | 3.8 |
| particlefilter | 50.9 | 0.4 | 0.4 | exe err | 1.0 |
| particles | 0.6 | 0.3 | 0.3 | 0.6 | exe err |
| pathfinder | 8.2 | 0.7 | 0.7 | exe err | 1.3 |
| pcc | 4.7 | 8.7 | 8.7 | -- | -- |
| perlin | 8.7 | 9.0 | 9.0 | -- | -- |
| permutate | 2.9 | 2.8 | 2.8 | 3.0 | 3.5 |
| permute | 5.9 | 0.4 | 0.4 | 5.2 | 1.0 |
| perplexity | 12.8 | 0.3 | 0.3 | 12.8 | 0.9 |
| phmm | 0.6 | 0.3 | 0.3 | 0.7 | 1.0 |
| pingpong | build err | 0.4 | 0.4 | -- | -- |
| pitch | 0.7 | 0.4 | 0.4 | -- | -- |
| pnpoly | 0.9 | 0.7 | 0.7 | 0.9 | 1.2 |
| pns | 1.3 | 1.0 | 1.0 | exe err | exe err |
| pointwise | 16.1 | 39.1 | 38.8 | 21.2 | exe err |
| pool | 8.2 | 2.9 | 2.9 | 8.2 | 3.5 |
| popcount | 23.4 | 0.5 | 0.5 | 23.4 | 1.1 |
| prefetch | 1.1 | 0.8 | 0.8 | -- | -- |
| present | 0.6 | 0.3 | 0.3 | 0.6 | 0.8 |
| prna | 4.8 | 4.7 | 4.7 | exe err | 5.3 |
| projectile | 0.9 | 0.7 | 0.7 | 0.9 | 1.3 |
| pso | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| qem | 0.7 | 7.0 | 6.7 | -- | -- |
| qkv | 0.8 | 1.1 | 1.1 | -- | -- |
| qrg | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| qtclustering | 0.8 | 0.3 | 0.3 | exe err | exe err |
| quicksort | 0.8 | 0.6 | 0.6 | exe err | 1.2 |
| radixsort | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| radixsort2 | 14.2 | 9.0 | 9.0 | -- | -- |
| rainflow | 49.0 | 2.7 | 2.7 | 57.9 | 3.8 |
| randomAccess | 1.1 | 0.8 | 0.8 | 1.1 | 1.4 |
| rayleighBenardConvection | 0.7 | 0.4 | 0.4 | -- | -- |
| reaction | 0.6 | 0.4 | 0.4 | 0.7 | 1.0 |
| recursiveGaussian | 0.6 | 0.3 | 0.3 | exe err | 0.9 |
| relu | 13.9 | 0.4 | build err | -- | -- |
| remap | 13.9 | 0.7 | 1.1 | -- | -- |
| resize | 48.4 | 4.7 | 4.7 | exe err | 5.1 |
| resnet-kernels | 0.0 | 0.0 | 0.0 | -- | -- |
| reverse | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| reverse2D | 25.0 | 1.4 | 1.4 | -- | -- |
| rfs | 48.2 | 2.3 | 2.3 | exe err | 2.9 |
| ring | 16.9 | 0.6 | 0.6 | -- | -- |
| rle | 25.7 | 0.3 | build err | -- | -- |
| rng-wallace | 0.8 | 0.6 | 0.6 | 0.8 | 1.2 |
| rodrigues | 54.0 | 0.3 | 0.3 | 61.6 | 0.9 |
| romberg | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| rotary | 0.7 | 0.4 | 0.4 | -- | -- |
| rowwiseMoments | 51.2 | 1.4 | 1.4 | -- | -- |
| rsbench | 0.6 | 0.4 | 0.4 | exe err | 0.9 |
| rsc | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| rsmt | 0.9 | -- | build err | -- | -- |
| rtm8 | 1.5 | 1.3 | 1.3 | 1.5 | 1.9 |
| rushlarsen | 0.0 | 0.3 | 0.4 | 0.0 | 0.9 |
| s3d | 0.9 | 0.3 | 0.3 | 1.2 | 0.9 |
| s8n | 6.1 | 1.1 | 1.1 | 6.1 | 1.7 |
| sa | 38.8 | 0.5 | 0.5 | -- | -- |
| sad | 0.6 | 0.3 | 0.3 | 0.0 | 0.9 |
| sampling | 0.7 | 0.6 | 0.6 | 1.0 | 1.1 |
| saxpy-ompt | exe err | 2.1 | 2.1 | -- | -- |
| sc | 0.6 | 0.3 | build err | -- | -- |
| scan | 15.8 | 4.6 | 4.6 | 50.1 | 5.2 |
| scan2 | 50.1 | 0.6 | 0.6 | 0.8 | 1.2 |
| scan3 | 0.8 | 0.6 | 0.6 | -- | -- |
| scel | 3.6 | 3.5 | 3.5 | 0.0 | 4.1 |
| score | 2.6 | 2.4 | exe err | -- | -- |
| sddmm-batch | 1.6 | 0.6 | 0.6 | -- | -- |
| seam-carving | 0.6 | 0.3 | 0.3 | -- | -- |
| secp256k1 | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| segment-reduce | 51.1 | 2.6 | 2.6 | -- | -- |
| segsort | 1.3 | 0.8 | build err | -- | -- |
| sheath | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| shmembench | 0.6 | 0.3 | 0.3 | 0.0 | 0.9 |
| shuffle | 1.6 | 1.4 | build err | -- | -- |
| si | build err | build err | build err | -- | -- |
| simpleMultiDevice | 0.7 | 0.3 | 0.3 | -- | -- |
| simpleSpmv | 0.0 | 0.7 | build err | 0.0 | 1.5 |
| simplemoc | 15.4 | 0.4 | 0.4 | 15.5 | 1.0 |
| slit | build err | build err | build err | -- | -- |
| slu | build err | 4.5 | 4.5 | build err | 5.1 |
| snake | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| sobel | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| sobol | 4.4 | 4.3 | 4.3 | 4.4 | 4.9 |
| softmax | 16.6 | 17.1 | build err | 16.6 | 17.7 |
| softmax-fused | 1.8 | 5.3 | build err | -- | -- |
| softmax-online | 5.3 | 3.6 | build err | -- | -- |
| sort | 0.8 | 0.5 | 0.5 | 0.7 | exe err |
| sortKV | 0.7 | 0.3 | 0.3 | -- | -- |
| sosfil | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| sparkler | 0.6 | 3.5 | 3.5 | -- | -- |
| spaxpby | 0.0 | 3.5 | 3.5 | -- | -- |
| spd2s | 0.0 | 7.4 | 7.4 | -- | -- |
| spgeam | 0.6 | 0.3 | 0.3 | -- | -- |
| spgemm | 11.0 | 0.4 | 0.4 | -- | -- |
| sph | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| split | 15.8 | 0.6 | 0.6 | 8.2 | 1.0 |
| spm | 10.6 | 0.4 | 0.4 | 13.4 | 1.0 |
| spmm | 0.6 | 0.3 | 0.3 | -- | -- |
| spmv | 0.0 | 0.7 | 0.7 | -- | -- |
| spnnz | 7.2 | 7.2 | 7.2 | -- | -- |
| sps2d | 0.0 | 7.4 | 7.4 | -- | -- |
| spsm | 0.0 | 0.3 | build err | -- | -- |
| spsort | 1.2 | 0.8 | 0.8 | -- | -- |
| sptrsv | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| srad | 0.6 | 15.1 | 15.1 | 28.8 | 15.7 |
| ss | 0.6 | 4.6 | 0.3 | 0.6 | 0.9 |
| ssim | 0.6 | 0.3 | build err | -- | -- |
| sss | exe err | exe err | exe err | -- | -- |
| sssp | 0.6 | 0.3 | 0.3 | -- | -- |
| stddev | 8.7 | 4.6 | 4.6 | 8.7 | 5.2 |
| stencil1d | 4.7 | 4.6 | 4.6 | 4.7 | 67.5 |
| stencil3d | 52.4 | 24.3 | 12.1 | exe err | 9.9 |
| streamCreateCopyDestroy | 0.6 | 0.6 | 0.6 | -- | -- |
| streamOrderedAllocation | 0.9 | 47.9 | 49.9 | -- | -- |
| streamPriority | 2.6 | 42.1 | 2.7 | -- | -- |
| streamUM | 51.9 | 0.7 | 0.7 | -- | -- |
| streamcluster | 0.6 | 0.4 | 0.4 | exe err | exe err |
| stsg | build err | build err | build err | -- | -- |
| su3 | 4.4 | 1.0 | 1.0 | 8.2 | 2.2 |
| surfel | 8.2 | 0.3 | 0.3 | 8.2 | 0.9 |
| svd3x3 | 8.2 | 0.4 | 0.4 | 0.7 | 1.0 |
| sw4ck | 1.9 | 2.2 | 2.2 | exe err | 2.2 |
| swish | 15.8 | 0.5 | 0.5 | 0.0 | 1.1 |
| tensorAccessor | 7.0 | 0.6 | 0.6 | -- | -- |
| tensorT | 3.6 | 3.5 | 3.5 | 5.2 | 4.1 |
| testSNAP | 3.2 | 3.0 | 3.0 | 3.2 | 0.6 |
| thomas | 0.0 | 0.8 | 0.8 | 0.0 | 1.4 |
| threadfence | 6.0 | 0.3 | 0.3 | 6.0 | 0.9 |
| tissue | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| tonemapping | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| tpacf | 0.8 | 0.6 | build err | -- | -- |
| tqs | 0.6 | 0.3 | 0.3 | exe err | 0.9 |
| triad | 0.7 | 0.4 | 0.4 | 1.1 | 1.2 |
| tridiagonal | 29.2 | 2.3 | 2.3 | 29.2 | 2.9 |
| tsa | exe err | 0.3 | 0.3 | exe err | 0.9 |
| tsne | 0.6 | 0.3 | 0.3 | -- | -- |
| tsp | 0.6 | 0.3 | 0.3 | build err | 0.9 |
| unfold | 13.6 | 0.4 | 0.4 | -- | -- |
| urng | 0.6 | 0.3 | 0.3 | 0.6 | 0.6 |
| vanGenuchten | 6.2 | 1.0 | 1.0 | 6.7 | 1.6 |
| vmc | 0.6 | 0.3 | 0.3 | exe err | 0.9 |
| vol2col | 3.5 | 3.4 | 3.4 | 5.5 | exe err |
| vote | 0.6 | 0.3 | build err | -- | -- |
| voxelization | 0.6 | 0.4 | 0.4 | -- | -- |
| warpexchange | 31.8 | 0.8 | 0.8 | -- | -- |
| warpsort | 0.7 | 0.3 | build err | -- | -- |
| wedford | 49.7 | 8.9 | build err | -- | -- |
| winograd | 0.6 | 0.3 | 0.3 | 0.6 | 0.8 |
| wlcpow | 0.7 | 0.4 | 0.4 | 0.7 | 1.0 |
| wmma | 27.7 | 0.5 | build err | -- | -- |
| word2vec | 1.6 | 2.4 | 2.4 | -- | -- |
| wordcount | 1.6 | 2.4 | 2.4 | 1.7 | 3.0 |
| wsm5 | 0.6 | 0.4 | 0.4 | 0.7 | 0.9 |
| wyllie | 0.0 | 0.4 | 0.4 | 0.0 | 1.0 |
| xlqc | 0.6 | 0.3 | 0.3 | 0.8 | 1.2 |
| xsbench | 6.3 | 6.3 | 6.3 | 6.3 | 6.9 |
| zerocopy | 0.6 | 0.3 | 1.1 | -- | -- |
| zeropoint | 15.8 | 16.3 | 16.3 | 0.0 | 16.9 |
| zmddft | 1.6 | 1.4 | 1.4 | 1.6 | 2.0 |
| zoom | 16.7 | 0.6 | 0.5 | -- | -- |
