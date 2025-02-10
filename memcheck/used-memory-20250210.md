# used memory [GB]

| name | cuda | hip | hipified | omp_nvc | omp_aomp |
| -- | -- | -- | -- | -- | -- |
| accuracy | | | | | |
| ace | | | | | exe err |
| adam | | | | | exe err |
| addBiasResidualLayerNorm | | | exe err | -- | -- |
| adv | | | | | |
| aes | | | | | |
| affine | | 0.3 | 0.3 | | 0.9 |
| aidw | | | | | |
| aligned-types | | | | | |
| all-pairs-distance | | | | | |
| allreduce | | | | -- | -- |
| amgmk | | | | | |
| ans | | | | exe err | |
| aobench | | | | | |
| aop | | | build err | | build err |
| asmooth | | | | | |
| assert | | exe err | exe err | -- | -- |
| asta | | | | | exe err |
| atan2 | | | | | |
| atomicAggregate | | | build err | -- | -- |
| atomicCAS | | | | -- | -- |
| atomicCost | | | | | |
| atomicIntrinsics | | | | | |
| atomicPerf | | | | | |
| atomicReduction | | | | | |
| atomicSystemWide | | | | -- | -- |
| attention | | | | | |
| attentionMultiHead | | | build err | -- | -- |
| axhelm | | | | | |
| b+tree | | | | | |
| babelstream | | | | | |
| background-subtract | | | | | |
| backprop | | | | | |
| bezier-surface | | | | | |
| bfs | | | | | |
| bh | | | | -- | -- |
| bicgstab | 0.7 | build err | | -- | -- |
| bilateral | | | | | |
| bincount | | | | -- | -- |
| binomial | | | | -- | |
| bitcracker | | | | -- | -- |
| bitonic-sort | | | | | |
| bitpacking | | | | -- | -- |
| bitpermute | | | build err | -- | -- |
| black-scholes | | | | | |
| blas-dot | 4.7 | 4.6 | 4.6 | -- | -- |
| blas-fp8gemm | | 1.1 | build err | -- | -- |
| blas-gemm | | | build err | -- | -- |
| blas-gemmBatched | | | build err | -- | -- |
| blas-gemmEx | | | | -- | -- |
| blas-gemmEx2 | | | | -- | -- |
| blas-gemmStridedBatched | | | | -- | -- |
| blockAccess | | | | -- | -- |
| blockexchange | | build err | build err | -- | -- |
| bm3d | exe err | | | -- | -- |
| bmf | | | build err | -- | -- |
| bn | | exe err | exe err | exe err | |
| bonds | | | | | |
| boxfilter | | | build err | | |
| bscan | | | | -- | -- |
| bsearch | | | | exe err | |
| bspline-vgh | | | | | |
| bsw | | | build err | -- | -- |
| btree | | -- | build err | -- | -- |
| burger | | | | | |
| bwt | | | | exe err | |
| car | | | | | |
| cbsfil | | | | | |
| cc | | | build err | -- | -- |
| ccl | | | | -- | -- |
| ccs | | | | | |
| ccsd-trpdrv | | | | | |
| ced | | | | -- | -- |
| cfd | | | | | |
| chacha20 | | | | | |
| channelShuffle | | | | | |
| channelSum | | | | | |
| che | | | | | |
| chemv | | | | | |
| chi2 | | | | | |
| clenergy | | | | | |
| clink | | | | | |
| clock | | | | -- | -- |
| cm | | | | build err | build err |
| cmembench | | | | -- | -- |
| cmp | | | | | build err |
| cobahh | 55.9 | 58.3 | 58.3 | 55.9 | 58.9 |
| collision | | | build err | -- | -- |
| colorwheel | | | | | |
| columnarSolver | | build err | build err | | build err |
| complex | | | | | |
| compute-score | | | | | |
| concat | | | | | |
| concurrentKernels | | | | -- | -- |
| contract | | | | | |
| conversion | | | build err | | |
| convolution1D | | | | | |
| convolution3D | | | build err | | build err |
| convolutionDeformable | | | | -- | -- |
| convolutionSeparable | | | | | |
| cooling | | | | | |
| coordinates | | | | -- | -- |
| copy | | | | -- | -- |
| crc64 | | | | | |
| cross | exe err | | | exe err | |
| crossEntropy | | | | -- | -- |
| crs | | | | | |
| d2q9-bgk | | | | | |
| d3q19-bgk | | | | -- | -- |
| damage | | | | | |
| daphne | | | | -- | -- |
| dct8x8 | | | | exe err | |
| ddbp | | | | | |
| debayer | | | | exe err | |
| degrid | | | | | |
| dense-embedding | | | | | |
| depixel | | | | | |
| deredundancy | | | | | |
| determinant | | | | -- | -- |
| diamond | exe err | | | exe err | |
| dispatch | | | | -- | -- |
| distort | | | | | |
| divergence | | | | | |
| doh | | | | | |
| dp | | | | | |
| dpid | | | | -- | -- |
| dropout | | | | -- | -- |
| dslash | | | | | |
| dwconv | | | | -- | -- |
| dwconv1d | | | | -- | -- |
| dxtc1 | | | | -- | -- |
| dxtc2 | | | | | exe err |
| easyWave | | | | | |
| ecdh | | | | | |
| egs | | | | -- | -- |
| eigenvalue | | | | | |
| eikonal | | | | -- | -- |
| entropy | | | | | |
| epistasis | | | | | |
| ert | | | | -- | -- |
| expdist | | | | | |
| extend2 | | | | | |
| extrema | | | | | |
| f16max | | | | -- | -- |
| f16sp | | | | -- | -- |
| face | | | | | exe err |
| fdtd3d | | | | | |
| feynman-kac | | | | | |
| fft | | | | | |
| fhd | | | | | |
| filter | | | | | |
| flame | | | | -- | -- |
| flip | | | | | |
| floydwarshall | | | | | |
| floydwarshall2 | | | | -- | -- |
| fluidSim | | | | | |
| fpc | | | | exe err | |
| fpdc | | | | | exe err |
| frechet | | | | exe err | |
| fresnel | | | | | |
| frna | | build err | build err | build err | build err |
| fsm | | | | | |
| fwt | | | | | |
| ga | | | | | |
| gabor | | | | exe err | |
| gamma-correction | | | | | |
| gaussian | | | | | |
| gc | | exe err | | | |
| gd | | | | | |
| ge-spmm | | | | -- | -- |
| geam | | | | -- | -- |
| gels | | | | -- | -- |
| gelu | | | | -- | -- |
| gemv | | | | -- | -- |
| geodesic | | | | | |
| gerbil | | | build err | -- | -- |
| gibbs | | | | -- | -- |
| glu | | | | | |
| gmm | | | | | |
| goulash | | | | | |
| gpp | | | | | |
| graphB+ | | | | -- | -- |
| graphExecution | | | | -- | -- |
| grep | | | | | |
| grrt | | | | | |
| gru | | | | -- | -- |
| haccmk | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| halo-finder | 0.6 | 0.3 | 0.3 | -- | -- |
| hausdorff | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| haversine | 1.0 | 0.8 | 0.8 | 1.0 | 1.4 |
| hbc | 0.6 | 0.3 | 0.3 | -- | -- |
| heartwall | 3.3 | 0.3 | 0.3 | exe err | exe err |
| heat | 16.9 | 17.5 | 17.5 | 16.9 | 18.1 |
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
| hybridsort | 1.2 | 0.9 | 0.7 | 1.2 | 1.5 |
| hypterm | 3.5 | 3.3 | 3.3 | 3.5 | 3.9 |
| idivide | 0.6 | 0.3 | 0.3 | 0.6 | exe err |
| interleave | 0.6 | 0.3 | 0.3 | 0.6 | 0.8 |
| interval | 1.0 | 0.6 | 0.6 | 1.0 | 1.2 |
| intrinsics-cast | 43.3 | 45.1 | 45.1 | -- | -- |
| intrinsics-simd | 15.8 | -- | build err | -- | -- |
| inversek2j | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| is | 1.7 | 1.4 | 1.4 | -- | -- |
| ising | 8.1 | 8.1 | 8.1 | 8.1 | 8.8 |
| iso2dfd | 49.7 | 51.8 | 51.8 | 49.7 | 52.4 |
| jaccard | 2.9 | 4.2 | build err | -- | -- |
| jacobi | 0.6 | 0.3 | build err | 0.6 | 0.9 |
| jenkins-hash | 10.0 | 10.2 | 10.2 | 10.0 | 10.8 |
| kalman | 10.8 | 11.2 | 11.2 | exe err | 11.4 |
| keccaktreehash | 0.7 | 0.4 | 0.4 | 0.7 | 1.0 |
| keogh | 3.6 | 3.5 | 3.5 | 5.8 | 5.7 |
| kernelLaunch | 0.6 | 0.3 | 0.3 | 0.6 | 0.8 |
| kmc | 0.7 | 0.4 | 0.4 | -- | -- |
| kmeans | 0.7 | 0.4 | 0.4 | 0.7 | 1.0 |
| knn | 0.6 | 0.4 | 0.4 | 0.6 | 1.0 |
| kurtosis | 48.6 | 50.7 | 50.7 | -- | -- |
| lanczos | 0.6 | 0.4 | 0.4 | 0.6 | 1.0 |
| langevin | 31.1 | 32.3 | 32.3 | 31.1 | 32.9 |
| langford | 2.1 | 1.9 | 1.9 | build err | exe err |
| laplace | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| laplace3d | 13.7 | 13.2 | 8.9 | 8.8 | 9.4 |
| lavaMD | 49.9 | 52.1 | 52.1 | 49.9 | 52.7 |
| layernorm | 0.6 | 0.3 | build err | -- | -- |
| layout | 0.6 | 0.4 | 0.4 | 0.6 | 0.9 |
| lci | 0.6 | 0.3 | 0.3 | 0.0 | 0.3 |
| lda | 0.6 | 0.4 | 0.4 | build err | 1.0 |
| ldpc | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| lebesgue | 9.7 | 68.2 | 68.3 | 50.9 | 67.9 |
| leukocyte | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| lfib4 | 8.7 | 8.9 | 8.9 | -- | -- |
| libor | 4.0 | 1.8 | 1.8 | 4.0 | 2.4 |
| lid-driven-cavity | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| lif | 23.4 | 24.3 | 24.3 | 23.4 | 24.9 |
| linearprobing | 1.4 | 0.9 | 0.8 | build err | exe err |
| log2 | 1.3 | 1.1 | 1.1 | exe err | 1.7 |
| logan | 2.7 | 2.5 | 2.6 | -- | -- |
| logic-resim | 0.8 | 0.7 | 0.7 | -- | -- |
| logic-rewrite | 19.4 | build err | build err | -- | -- |
| logprob | 8.4 | 8.5 | build err | -- | -- |
| lombscargle | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| loopback | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| lr | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| lrn | 22.5 | 23.3 | 23.3 | 25.5 | 23.9 |
| lsqt | build err | 1.8 | build err | 2.0 | 2.3 |
| lud | 4.7 | 4.6 | 4.6 | 4.7 | 5.2 |
| ludb | 0.7 | 0.4 | build err | -- | -- |
| lulesh | 54.2 | 56.6 | 56.6 | 54.2 | 13.0 |
| lzss | | | | -- | -- |
| mallocFree | 8.5 | 13.8 | build err | 16.6 | 7.9 |
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
| merge | 31.1 | 32.3 | 32.3 | 46.3 | 32.9 |
| merkle | 1.6 | 1.6 | 1.6 | -- | -- |
| metropolis | | | build err | | |
| mf-sgd | | | | -- | -- |
| michalewicz | | | | | build err |
| miniDGS | build err | -- | | -- | -- |
| miniFE | | | | build err | build err |
| miniWeather | | build err | build err | | |
| minibude | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| minimap2 | 1.0 | build err | 0.7 | 1.0 | 1.6 |
| minimod | | | | -- | -- |
| minisweep | | | | build err | |
| minkowski | | | | | |
| minmax | 51.0 | 54.1 | 54.1 | -- | -- |
| mis | | | | build err | |
| mixbench | | build err | | | build err |
| mmcsf | | | | -- | -- |
| mnist | | | | -- | -- |
| morphology | | | | | |
| mpc | 1.1 | | exe err | -- | -- |
| mr | | | | build err | |
| mrc | | | | | build err |
| mrg32k3a | 6.0 | 11.6 | 11.6 | -- | -- |
| mriQ | 0.6 | 0.3 | 0.3 | 0.6 | 0.9 |
| mt | 0.7 | 0.4 | 0.4 | 0.7 | 1.0 |
| mtf | | | | -- | -- |
| multimaterial | 3.1 | 2.9 | 2.9 | 3.1 | 3.5 |
| multinomial | | | | -- | -- |
| murmurhash3 | 2.6 | 2.4 | 2.4 | 2.6 | 0.0 |
| myocyte | | | | | build err |
| nbnxm | | | | -- | -- |
| nbody | | | | | |
| ne | | | | | build err |
| nlll | | | | | build err |
| nms | | | | build err | build err |
| nn | | | | | |
| nonzero | | | | -- | -- |
| norm2 | | build err | build err | | build err |
| nosync | | | build err | -- | -- |
| nqueen | | | | | |
| ntt | | | | | |
| nw | | | | | |
| openmp | | | | | |
| opticalFlow | | | build err | -- | -- |
| overlap | | | build err | -- | -- |
| overlay | | | | | build err |
| p2p | | | | -- | -- |
| p4 | | | | | |
| pad | | | build err | -- | -- |
| page-rank | | | | | build err |
| particle-diffusion | | | | | |
| particlefilter | | | | build err | |
| particles | | | | | build err |
| pathfinder | | | | build err | |
| pcc | | build err | build err | -- | -- |
| perlin | | | | -- | -- |
| permutate | | | | build err | |
| permute | | | build err | | |
| perplexity | | | | | build err |
| phmm | | | | | |
| pingpong | | build err | build err | -- | -- |
| pitch | | | | -- | -- |
| pnpoly | | | | | build err |
| pns | | | | build err | |
| pointwise | | | | | build err |
| pool | | | | | |
| popcount | | | | | |
| prefetch | | | | -- | -- |
| present | | | | | |
| prna | | | build err | build err | |
| projectile | | | | | |
| pso | | | | | |
| qem | | | build err | -- | -- |
| qkv | | | build err | -- | -- |
| qrg | | | | | build err |
| qtclustering | | | | build err | build err |
| quicksort | | | | build err | build err |
| radixsort | | | | | build err |
| radixsort2 | | | | -- | -- |
| rainflow | | | | | |
| randomAccess | build err | | | | |
| rayleighBenardConvection | | | build err | -- | -- |
| reaction | | | | | |
| recursiveGaussian | | | build err | build err | build err |
| relu | | | build err | -- | -- |
| remap | | | | -- | -- |
| resize | | | | build err | build err |
| resnet-kernels | | | | -- | -- |
| reverse | | | | | build err |
| reverse2D | | | | -- | -- |
| rfs | | | | build err | build err |
| ring | | | | -- | -- |
| rle | | | build err | -- | -- |
| rng-wallace | | | | | build err |
| rodrigues | | | | | |
| romberg | | | | | build err |
| rotary | | | build err | -- | -- |
| rowwiseMoments | | | | -- | -- |
| rsbench | | | | build err | |
| rsc | | | | build err | build err |
| rsmt | | -- | | -- | -- |
| rtm8 | | | | | |
| rushlarsen | | | | | |
| s3d | | | | | |
| s8n | | | | | |
| sa | | | | -- | -- |
| sad | | | | build err | |
| sampling | | | | | |
| saxpy-ompt | exe err | | | -- | -- |
| sc | | | build err | -- | -- |
| scan | | | | | |
| scan2 | | | | | |
| scan3 | | | | -- | -- |
| scel | | | | | build err |
| score | | | build err | -- | -- |
| sddmm-batch | | | build err | -- | -- |
| seam-carving | | | | -- | -- |
| secp256k1 | | | | | |
| segment-reduce | | | | -- | -- |
| segsort | | | build err | -- | -- |
| sheath | | | | | build err |
| shmembench | | | | build err | |
| shuffle | | | build err | -- | -- |
| si | | | | -- | -- |
| simpleMultiDevice | | | | -- | -- |
| simpleSpmv | | | build err | | |
| simplemoc | | | | | |
| slit | | | | -- | -- |
| slu | | | | | |
| snake | | | | | |
| sobel | | | | | |
| sobol | | | | | |
| softmax | | | build err | build err | build err |
| softmax-fused | | | build err | -- | -- |
| softmax-online | | | build err | -- | -- |
| sort | | | | | build err |
| sortKV | | | | -- | -- |
| sosfil | | | | | build err |
| sparkler | | build err | build err | -- | -- |
| spaxpby | | build err | build err | -- | -- |
| spd2s | | | build err | -- | -- |
| spgeam | | | build err | -- | -- |
| spgemm | | | build err | -- | -- |
| sph | | | | | build err |
| split | | | | | build err |
| spm | | | | | build err |
| spmm | | | build err | -- | -- |
| spmv | | | build err | -- | -- |
| spnnz | | | build err | -- | -- |
| sps2d | | | build err | -- | -- |
| spsm | | | build err | -- | -- |
| spsort | | | build err | -- | -- |
| sptrsv | | | | | build err |
| srad | | | | build err | build err |
| ss | | | | | |
| ssim | | | build err | -- | -- |
| sss | | | build err | -- | -- |
| sssp | | | | -- | -- |
| stddev | | | | | build err |
| stencil1d | | | | build err | |
| stencil3d | | | | build err | |
| streamCreateCopyDestroy | | | | -- | -- |
| streamOrderedAllocation | | | | -- | -- |
| streamPriority | | | | -- | -- |
| streamUM | | | build err | -- | -- |
| streamcluster | | | | | build err |
| stsg | build err | build err | build err | -- | -- |
| su3 | | | | | build err |
| surfel | | | | | |
| svd3x3 | | | | | |
| sw4ck | | | | build err | |
| swish | | | | | build err |
| tensorAccessor | | | | -- | -- |
| tensorT | | | | | |
| testSNAP | | | | | |
| thomas | | | | | |
| threadfence | | | | | build err |
| tissue | | | | | build err |
| tonemapping | | | | | |
| tpacf | | | build err | -- | -- |
| tqs | | | | build err | build err |
| triad | | | | | build err |
| tridiagonal | | | | | build err |
| tsa | | | | | build err |
| tsne | build err | build err | build err | -- | -- |
| tsp | | | | build err | build err |
| unfold | | | build err | -- | -- |
| urng | | | | | |
| vanGenuchten | | | | | build err |
| vmc | | | | build err | build err |
| vol2col | | | | | build err |
| vote | | | build err | -- | -- |
| voxelization | | | | -- | -- |
| warpexchange | | | build err | -- | -- |
| warpsort | | | build err | -- | -- |
| wedford | | | build err | -- | -- |
| winograd | | | | | |
| wlcpow | | | | | build err |
| wmma | | | build err | -- | -- |
| word2vec | | | | -- | -- |
| wordcount | | | | | |
| wsm5 | | | | | build err |
| wyllie | | | | | |
| xlqc | build err | | | build err | |
| xsbench | | | | | |
| zerocopy | | | | -- | -- |
| zeropoint | 15.8 | 16.3 | 16.3 | 22.4 | 16.9 |
| zmddft | | | | | |
| zoom | | | | -- | -- |
