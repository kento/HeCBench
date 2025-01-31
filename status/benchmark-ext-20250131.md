| name | cuda | hip | hipified | omp_nvc | omp_aomp | plot |
| -- | -- | -- | -- | -- | -- | -- |
| accuracy | 5.55 | 6.49 | 6.48 | 63.77 | 10.19 |![accuracy](SVGs/accuracy.svg) |
| ace | 9.52 | 3.35 | 3.32 | 5.15 | exe err |![ace](SVGs/ace.svg) |
| adam | 7.08 | 4.20 | 4.19 | 8.65 | exe err |![adam](SVGs/adam.svg) |
| addBiasResidualLayerNorm | 18.92 | 50.36 | exe err | -- | -- |![addBiasResidualLayerNorm](SVGs/addBiasResidualLayerNorm.svg) |
| adv | 7.32 | 12.87 | 13.06 | 7.81 | 4.22 |![adv](SVGs/adv.svg) |
| aes | 8.95 | 1.23 | 1.21 | 8.99 | 0.74 |![aes](SVGs/aes.svg) |
| affine | 7.76 | 4.76 | 4.77 | 14.42 | |![affine](SVGs/affine.svg) |
| aidw | 11.02 | 9.75 | 48.61 | 11.46 | 87.89 |![aidw](SVGs/aidw.svg) |
| aligned-types | 5.70 | 2.16 | 2.16 | 6.91 | 3.42 |![aligned-types](SVGs/aligned-types.svg) |
| all-pairs-distance | 6.19 | 22.04 | 21.94 | 20.97 | 3.41 |![all-pairs-distance](SVGs/all-pairs-distance.svg) |
| allreduce | | | | -- | -- |![allreduce](SVGs/allreduce.svg) |
| amgmk | 4.53 | 0.75 | 0.74 | 4.52 | 0.80 |![amgmk](SVGs/amgmk.svg) |
| ans | 7.25 | | | 6.10 | |![ans](SVGs/ans.svg) |
| aobench | 4.48 | 0.48 | 0.48 | 7.60 | 1.53 |![aobench](SVGs/aobench.svg) |
| aop | 8.77 | 34.59 | | 306.19 | 0.86 |![aop](SVGs/aop.svg) |
| asmooth | 18.28 | 12.71 | 13.61 | 13.83 | |![asmooth](SVGs/asmooth.svg) |
| assert | 4.36 | exe err | exe err | -- | -- |![assert](SVGs/assert.svg) |
| asta | 6.24 | 3.67 | 3.69 | 12.83 | 0.34 |![asta](SVGs/asta.svg) |
| atan2 | 5.30 | 0.76 | 0.76 | 5.37 | |![atan2](SVGs/atan2.svg) |
| atomicAggregate | 7.63 | 5.48 | | -- | -- |![atomicAggregate](SVGs/atomicAggregate.svg) |
| atomicCAS | 17.72 | 14.55 | 15.02 | -- | -- |![atomicCAS](SVGs/atomicCAS.svg) |
| atomicCost | 97.45 | 7.86 | 7.93 | 96.88 | 25.15 |![atomicCost](SVGs/atomicCost.svg) |
| atomicIntrinsics | 89.73 | 70.29 | 70.31 | 4.58 | 0.90 |![atomicIntrinsics](SVGs/atomicIntrinsics.svg) |
| atomicPerf | 11.11 | 5.26 | | 12.14 | 372.30 |![atomicPerf](SVGs/atomicPerf.svg) |
| atomicReduction | 5.63 | 0.92 | 0.92 | | 1.79 |![atomicReduction](SVGs/atomicReduction.svg) |
| atomicSystemWide | 5.31 | 10.19 | 11.58 | -- | -- |![atomicSystemWide](SVGs/atomicSystemWide.svg) |
| attention | 40.64 | 158.70 | | 0.01 | |![attention](SVGs/attention.svg) |
| attentionMultiHead | 5.36 | 1.66 | build err | -- | -- |![attentionMultiHead](SVGs/attentionMultiHead.svg) |
| axhelm | 5.01 | | 0.81 | 10.07 | 0.73 |![axhelm](SVGs/axhelm.svg) |
| b+tree | 4.89 | 0.96 | 0.94 | 5.40 | 0.72 |![b+tree](SVGs/b+tree.svg) |
| babelstream | 4.64 | 0.95 | 0.95 | 4.65 | 3.11 |![babelstream](SVGs/babelstream.svg) |
| background-subtract | 5.70 | 4.74 | 4.73 | 7.43 | |![background-subtract](SVGs/background-subtract.svg) |
| backprop | 4.51 | 0.46 | 0.46 | 4.94 | 0.25 |![backprop](SVGs/backprop.svg) |
| bezier-surface | 18.36 | 23.79 | 23.80 | 17.16 | |![bezier-surface](SVGs/bezier-surface.svg) |
| bfs | 5.32 | 1.18 | 1.12 | 5.31 | 0.95 |![bfs](SVGs/bfs.svg) |
| bh | 8.16 | | | -- | -- |![bh](SVGs/bh.svg) |
| bicgstab | 5.17 | | | -- | -- |![bicgstab](SVGs/bicgstab.svg) |
| bilateral | 9.52 | 46.72 | 18.22 | 10.11 | |![bilateral](SVGs/bilateral.svg) |
| bincount | 5.48 | 2.01 | 2.01 | -- | -- |![bincount](SVGs/bincount.svg) |
| binomial | 4.92 | 1.78 | 1.79 | -- | |![binomial](SVGs/binomial.svg) |
| bitcracker | exe err | exe err | exe err | -- | -- |![bitcracker](SVGs/bitcracker.svg) |
| bitonic-sort | 16.45 | 13.42 | 13.35 | 11.35 | |![bitonic-sort](SVGs/bitonic-sort.svg) |
| bitpacking | 8.14 | 4.10 | 4.09 | -- | -- |![bitpacking](SVGs/bitpacking.svg) |
| bitpermute | 6.01 | 7.11 | | -- | -- |![bitpermute](SVGs/bitpermute.svg) |
| black-scholes | 9.55 | 3.97 | 3.95 | 7.78 | 8.32 |![black-scholes](SVGs/black-scholes.svg) |
| blas-dot | 17.29 | 9.95 | | -- | -- |![blas-dot](SVGs/blas-dot.svg) |
| blas-fp8gemm | 4.52 | | | -- | -- |![blas-fp8gemm](SVGs/blas-fp8gemm.svg) |
| blas-gemm | 12.53 | 4.51 | | -- | -- |![blas-gemm](SVGs/blas-gemm.svg) |
| blas-gemmBatched | 19.85 | 21.61 | | -- | -- |![blas-gemmBatched](SVGs/blas-gemmBatched.svg) |
| blas-gemmEx | 15.99 | | | -- | -- |![blas-gemmEx](SVGs/blas-gemmEx.svg) |
| blas-gemmEx2 | 16.02 | | | -- | -- |![blas-gemmEx2](SVGs/blas-gemmEx2.svg) |
| blas-gemmStridedBatched | 20.18 | 21.54 | | -- | -- |![blas-gemmStridedBatched](SVGs/blas-gemmStridedBatched.svg) |
| blockAccess | 5.76 | 8.15 | 8.13 | -- | -- |![blockAccess](SVGs/blockAccess.svg) |
| blockexchange | 33.67 | | | -- | -- |![blockexchange](SVGs/blockexchange.svg) |
| bm3d | 0.32 | 0.06 | 0.04 | -- | -- |![bm3d](SVGs/bm3d.svg) |
| bmf | 5.99 | 7.26 | | -- | -- |![bmf](SVGs/bmf.svg) |
| bn | 6.33 | 10.18 | 10.23 | 6.72 | |![bn](SVGs/bn.svg) |
| bonds | 14.44 | 19.26 | 19.28 | 16.35 | 15.18 |![bonds](SVGs/bonds.svg) |
| boxfilter | 8.75 | 9.19 | | 11.78 | 37.43 |![boxfilter](SVGs/boxfilter.svg) |
| bscan | 10.62 | 1.78 | 0.44 | -- | -- |![bscan](SVGs/bscan.svg) |
| bsearch | 743.47 | 4.61 | 4.61 | 72.35 | 34.57 |![bsearch](SVGs/bsearch.svg) |
| bspline-vgh | 14.11 | 10.08 | 10.08 | 9.98 | 11.30 |![bspline-vgh](SVGs/bspline-vgh.svg) |
| bsw | 4.56 | 0.56 | | -- | -- |![bsw](SVGs/bsw.svg) |
| btree | 157.30 | -- | | -- | -- |![btree](SVGs/btree.svg) |
| burger | 110.83 | 21.49 | 21.50 | 75.83 | |![burger](SVGs/burger.svg) |
| bwt | 5904.43 | 13.38 | 13.66 | 5837.90 | 15.18 |![bwt](SVGs/bwt.svg) |
| car | 11.59 | 20.00 | 20.00 | 11.59 | |![car](SVGs/car.svg) |
| cbsfil | 43.47 | 5.94 | 5.94 | 42.67 | |![cbsfil](SVGs/cbsfil.svg) |
| cc | 3.49 | 2.48 | build err | -- | -- |![cc](SVGs/cc.svg) |
| ccl | | | | -- | -- |![ccl](SVGs/ccl.svg) |
| ccs | 5.47 | 6.11 | 6.10 | 7.24 | 0.27 |![ccs](SVGs/ccs.svg) |
| ccsd-trpdrv | 41.53 | 19.17 | 19.35 | 35.93 | 2.34 |![ccsd-trpdrv](SVGs/ccsd-trpdrv.svg) |
| ced | 7.05 | 3.23 | 3.21 | -- | -- |![ced](SVGs/ced.svg) |
| cfd | 5.10 | 1.51 | 1.48 | 5.44 | 9.98 |![cfd](SVGs/cfd.svg) |
| chacha20 | 6.78 | 8.25 | 8.24 | 8.89 | 7.34 |![chacha20](SVGs/chacha20.svg) |
| channelShuffle | 49.94 | 38.92 | 32.81 | 68.93 | 76.38 |![channelShuffle](SVGs/channelShuffle.svg) |
| channelSum | 66.15 | 52.19 | 52.10 | 69.99 | |![channelSum](SVGs/channelSum.svg) |
| che | 0.00 | 0.00 | 0.00 | | 0.00 |![che](SVGs/che.svg) |
| chemv | 5.08 | 1.90 | 1.91 | 5.54 | |![chemv](SVGs/chemv.svg) |
| chi2 | 223.36 | 20.43 | 20.42 | 260.91 | |![chi2](SVGs/chi2.svg) |
| clenergy | 4.83 | 1.34 | 1.34 | 5.26 | 17.58 |![clenergy](SVGs/clenergy.svg) |
| clink | 8.16 | 5.14 | 5.15 | 16.48 | 11.76 |![clink](SVGs/clink.svg) |
| clock | 4.48 | 0.44 | 0.42 | -- | -- |![clock](SVGs/clock.svg) |
| cm | 0.00 | 0.17 | 0.16 | | |![cm](SVGs/cm.svg) |
| cmembench | | | | -- | -- |![cmembench](SVGs/cmembench.svg) |
| cmp | | | | | |![cmp](SVGs/cmp.svg) |
| cobahh | 541.25 | | | | |![cobahh](SVGs/cobahh.svg) |
| collision | 9.90 | | | -- | -- |![collision](SVGs/collision.svg) |
| colorwheel | 48.47 | | | 42.53 | |![colorwheel](SVGs/colorwheel.svg) |
| columnarSolver | 11.55 | | | 13.73 | |![columnarSolver](SVGs/columnarSolver.svg) |
| complex | 25.25 | 3.68 | 3.67 | 15.22 | 26.06 |![complex](SVGs/complex.svg) |
| compute-score | 6.84 | 4.72 | 4.74 | 8.43 | |![compute-score](SVGs/compute-score.svg) |
| concat | 26.60 | 13.27 | 13.17 | 26.21 | 57.74 |![concat](SVGs/concat.svg) |
| concurrentKernels | 283.38 | 31.63 | 31.63 | -- | -- |![concurrentKernels](SVGs/concurrentKernels.svg) |
| contract | 1497.61 | 15.19 | 15.24 | 1418.37 | 24.22 |![contract](SVGs/contract.svg) |
| conversion | 6.77 | 0.94 | | 5.45 | 1.88 |![conversion](SVGs/conversion.svg) |
| convolution1D | 1715.16 | 100.98 | 100.98 | 3870.95 | |![convolution1D](SVGs/convolution1D.svg) |
| convolution3D | 2.86 | 83.03 | | 4.63 | |![convolution3D](SVGs/convolution3D.svg) |
| convolutionDeformable | | | | -- | -- |![convolutionDeformable](SVGs/convolutionDeformable.svg) |
| convolutionSeparable | 19.86 | 25.91 | 25.75 | 30.44 | |![convolutionSeparable](SVGs/convolutionSeparable.svg) |
| cooling | 2269.87 | 1.06 | 1.89 | 1115.12 | |![cooling](SVGs/cooling.svg) |
| coordinates | 510.74 | 3.35 | 3.34 | -- | -- |![coordinates](SVGs/coordinates.svg) |
| copy | 30.88 | 24.36 | | -- | -- |![copy](SVGs/copy.svg) |
| crc64 | 159.62 | 4.06 | 4.05 | 168.38 | 4.52 |![crc64](SVGs/crc64.svg) |
| cross | 12.96 | 11.32 | 11.43 | 387.76 | |![cross](SVGs/cross.svg) |
| crossEntropy | 11.71 | 10.41 | 10.40 | -- | -- |![crossEntropy](SVGs/crossEntropy.svg) |
| crs | 1259.70 | 10.30 | 10.30 | | 11.64 |![crs](SVGs/crs.svg) |
| d2q9-bgk | 5.74 | 1.48 | 1.48 | 7.58 | |![d2q9-bgk](SVGs/d2q9-bgk.svg) |
| d3q19-bgk | | 24.91 | 24.99 | -- | -- |![d3q19-bgk](SVGs/d3q19-bgk.svg) |
| damage | 25.55 | 1.02 | 1.02 | 222.52 | 0.79 |![damage](SVGs/damage.svg) |
| daphne | | | | -- | -- |![daphne](SVGs/daphne.svg) |
| dct8x8 | 181.81 | 1.50 | 1.51 | exe err | |![dct8x8](SVGs/dct8x8.svg) |
| ddbp | 18.57 | 4.88 | 4.88 | 18.44 | |![ddbp](SVGs/ddbp.svg) |
| debayer | 15.96 | 0.93 | 1.08 | exe err | 1.39 |![debayer](SVGs/debayer.svg) |
| degrid | 17.95 | 48.54 | | 22.38 | 115.32 |![degrid](SVGs/degrid.svg) |
| dense-embedding | 198.45 | 179.92 | 179.99 | 1777.93 | |![dense-embedding](SVGs/dense-embedding.svg) |
| depixel | 1163.31 | 30.94 | 37.48 | 6489.09 | |![depixel](SVGs/depixel.svg) |
| deredundancy | 18.36 | 59.29 | 59.26 | 18.47 | 13.02 |![deredundancy](SVGs/deredundancy.svg) |
| determinant | 4.73 | | | -- | -- |![determinant](SVGs/determinant.svg) |
| diamond | exe err | | | exe err | |![diamond](SVGs/diamond.svg) |
| dispatch | 6.37 | 1.98 | 2.02 | -- | -- |![dispatch](SVGs/dispatch.svg) |
| distort | 4.51 | 0.52 | 0.51 | 4.57 | 1.52 |![distort](SVGs/distort.svg) |
| divergence | 6.50 | 1.64 | 1.64 | 5.03 | 4.81 |![divergence](SVGs/divergence.svg) |
| doh | 18.82 | 19.75 | 19.75 | 6.18 | |![doh](SVGs/doh.svg) |
| dp | 54.96 | | | 34.07 | 31.61 |![dp](SVGs/dp.svg) |
| dpid | 50.59 | 1.02 | | -- | -- |![dpid](SVGs/dpid.svg) |
| dropout | 31.01 | 0.93 | 0.93 | -- | -- |![dropout](SVGs/dropout.svg) |
| dslash | 6.75 | 2.19 | 4.42 | 15.43 | 4.68 |![dslash](SVGs/dslash.svg) |
| dwconv | 81.64 | 23.33 | 23.22 | -- | -- |![dwconv](SVGs/dwconv.svg) |
| dwconv1d | | | | -- | -- |![dwconv1d](SVGs/dwconv1d.svg) |
| dxtc1 | 4.42 | 0.62 | 0.63 | -- | -- |![dxtc1](SVGs/dxtc1.svg) |
| dxtc2 | 4.53 | 0.66 | | 6.69 | 0.60 |![dxtc2](SVGs/dxtc2.svg) |
| easyWave | 7.11 | 2.48 | 2.46 | 7.25 | 3.17 |![easyWave](SVGs/easyWave.svg) |
| ecdh | 37.30 | 2.97 | 2.98 | 40.94 | 43.45 |![ecdh](SVGs/ecdh.svg) |
| egs | 5.07 | 0.17 | | -- | -- |![egs](SVGs/egs.svg) |
| eigenvalue | 23.90 | 13.44 | 13.40 | | 16.98 |![eigenvalue](SVGs/eigenvalue.svg) |
| eikonal | 206.17 | 10.50 | 9.93 | -- | -- |![eikonal](SVGs/eikonal.svg) |
| entropy | 57.12 | 10.30 | 10.31 | 57.16 | |![entropy](SVGs/entropy.svg) |
| epistasis | 67.92 | 189.71 | 190.18 | 136.80 | |![epistasis](SVGs/epistasis.svg) |
| ert | 7.38 | 3.53 | 3.53 | -- | -- |![ert](SVGs/ert.svg) |
| expdist | 9.41 | 7.22 | 7.25 | 15.58 | |![expdist](SVGs/expdist.svg) |
| extend2 | 7.56 | 11.35 | 11.35 | 6.81 | |![extend2](SVGs/extend2.svg) |
| extrema | 8.83 | 6.38 | 6.40 | 9.40 | 20.70 |![extrema](SVGs/extrema.svg) |
| f16max | 25.83 | 35.51 | 35.49 | -- | -- |![f16max](SVGs/f16max.svg) |
| f16sp | 33.71 | 30.75 | | -- | -- |![f16sp](SVGs/f16sp.svg) |
| face | 5.04 | 1.05 | 1.05 | | 0.69 |![face](SVGs/face.svg) |
| fdtd3d | 35.55 | 4.92 | 4.67 | 49.94 | 4.43 |![fdtd3d](SVGs/fdtd3d.svg) |
| feynman-kac | 84.36 | 113.98 | 115.15 | 24.12 | |![feynman-kac](SVGs/feynman-kac.svg) |
| fft | 6.52 | 2.18 | 2.18 | 7.49 | 0.49 |![fft](SVGs/fft.svg) |
| fhd | 765.79 | 1.11 | 1.09 | | |![fhd](SVGs/fhd.svg) |
| filter | 174.29 | 15.03 | 15.04 | 174.81 | 7.50 |![filter](SVGs/filter.svg) |
| flame | 5.83 | 3.90 | 3.93 | -- | -- |![flame](SVGs/flame.svg) |
| flip | 85.49 | 10.28 | 10.30 | 90.87 | 43.39 |![flip](SVGs/flip.svg) |
| floydwarshall | 78.88 | 1.92 | 1.95 | 97.80 | 5.67 |![floydwarshall](SVGs/floydwarshall.svg) |
| floydwarshall2 | 10.25 | 6.22 | | -- | -- |![floydwarshall2](SVGs/floydwarshall2.svg) |
| fluidSim | 22.99 | 25.92 | 25.54 | 21.98 | 30.36 |![fluidSim](SVGs/fluidSim.svg) |
| fpc | 70.35 | 1.40 | 1.40 | 30.19 | 0.80 |![fpc](SVGs/fpc.svg) |
| fpdc | 7.70 | 2.98 | 2.96 | | 2.19 |![fpdc](SVGs/fpdc.svg) |
| frechet | 15.26 | 1.11 | 1.11 | 144.67 | |![frechet](SVGs/frechet.svg) |
| fresnel | 8.65 | 6.64 | | | |![fresnel](SVGs/fresnel.svg) |
| frna | 0.00 | | | 0.00 | |![frna](SVGs/frna.svg) |
| fsm | 5.75 | 6.99 | 7.00 | | 0.34 |![fsm](SVGs/fsm.svg) |
| fwt | 5.36 | | 1.61 | 5.83 | |![fwt](SVGs/fwt.svg) |
| ga | 52.84 | 5.01 | 5.02 | 36.55 | 5.93 |![ga](SVGs/ga.svg) |
| gabor | 120.39 | 4.97 | 8.04 | exe err | |![gabor](SVGs/gabor.svg) |
| gamma-correction | 2178.54 | 19.22 | 19.21 | 3350.52 | 16.87 |![gamma-correction](SVGs/gamma-correction.svg) |
| gaussian | 20.74 | 5.89 | 5.67 | 9.21 | |![gaussian](SVGs/gaussian.svg) |
| gc | 4.56 | 0.51 | | | 0.14 |![gc](SVGs/gc.svg) |
| gd | 13.68 | 12.99 | 28.99 | 14.83 | 12.90 |![gd](SVGs/gd.svg) |
| ge-spmm | 7.54 | 8.63 | 8.59 | -- | -- |![ge-spmm](SVGs/ge-spmm.svg) |
| geam | 127.40 | | | -- | -- |![geam](SVGs/geam.svg) |
| gels | 4.67 | 1.83 | | -- | -- |![gels](SVGs/gels.svg) |
| gelu | 57.17 | 58.40 | 57.65 | -- | -- |![gelu](SVGs/gelu.svg) |
| gemv | 9.81 | 78.16 | | -- | -- |![gemv](SVGs/gemv.svg) |
| geodesic | exe err | exe err | exe err | exe err | exe err |![geodesic](SVGs/geodesic.svg) |
| gerbil | | | | -- | -- |![gerbil](SVGs/gerbil.svg) |
| gibbs | 4.53 | 0.71 | 0.70 | -- | -- |![gibbs](SVGs/gibbs.svg) |
| glu | 156.41 | 47.28 | 118.35 | 1593.95 | |![glu](SVGs/glu.svg) |
| gmm | 323.54 | 1.92 | 1.03 | 341.15 | |![gmm](SVGs/gmm.svg) |
| goulash | 142.59 | 11.60 | 11.66 | 147.07 | 20.07 |![goulash](SVGs/goulash.svg) |
| gpp | 6.83 | 238.43 | 234.16 | 6.99 | 29.99 |![gpp](SVGs/gpp.svg) |
| graphB+ | 7.21 | | | -- | -- |![graphB+](SVGs/graphB+.svg) |
| graphExecution | 6.60 | 1.84 | 1.84 | -- | -- |![graphExecution](SVGs/graphExecution.svg) |
| grep | 992.41 | 96.57 | 96.60 | 507.36 | 30.55 |![grep](SVGs/grep.svg) |
| grrt | 17.00 | 55.86 | 52.29 | 15.40 | |![grrt](SVGs/grrt.svg) |
| gru | 26.44 | 48.42 | 48.44 | -- | -- |![gru](SVGs/gru.svg) |
| haccmk | 6.54 | 4.25 | 4.25 | 6.38 | 5.41 |![haccmk](SVGs/haccmk.svg) |
| halo-finder | 5.44 | 4.50 | 7.77 | -- | -- |![halo-finder](SVGs/halo-finder.svg) |
| hausdorff | 32.73 | 14.40 | 14.40 | 33.92 | 15.64 |![hausdorff](SVGs/hausdorff.svg) |
| haversine | 6.46 | 1.82 | 2.05 | 6.52 | 2.09 |![haversine](SVGs/haversine.svg) |
| hbc | 15.68 | 15.25 | 15.24 | -- | -- |![hbc](SVGs/hbc.svg) |
| heartwall | 4.66 | 1.97 | 1.93 | exe err | 2.28 |![heartwall](SVGs/heartwall.svg) |
| heat | 26.31 | 39.72 | 40.25 | 91.48 | 69.52 |![heat](SVGs/heat.svg) |
| heat2d | build err | 11.03 | 11.50 | 248.52 | 329.62 |![heat2d](SVGs/heat2d.svg) |
| hellinger | 7.13 | 6.61 | 6.62 | 6.87 | 6.52 |![hellinger](SVGs/hellinger.svg) |
| henry | 5.87 | 2.89 | 4.51 | 5.07 | 3.89 |![henry](SVGs/henry.svg) |
| hexciton | 6.06 | 7.37 | build err | 6.82 | 17.83 |![hexciton](SVGs/hexciton.svg) |
| histogram | 10.43 | 1.30 | 1.33 | 5.30 | 8.38 |![histogram](SVGs/histogram.svg) |
| hmm | 7.18 | 3.94 | 3.97 | 17.31 | 7.21 |![hmm](SVGs/hmm.svg) |
| hogbom | 5.19 | 1.03 | 0.93 | 5.20 | 0.99 |![hogbom](SVGs/hogbom.svg) |
| hotspot | 4.71 | 0.60 | 0.59 | -- | -- |![hotspot](SVGs/hotspot.svg) |
| hotspot3D | 22.97 | 23.21 | 23.24 | 19.42 | 27.55 |![hotspot3D](SVGs/hotspot3D.svg) |
| hpl | 115.39 | 112.16 | build err | -- | -- |![hpl](SVGs/hpl.svg) |
| hungarian | 4.65 | 0.76 | 0.75 | -- | -- |![hungarian](SVGs/hungarian.svg) |
| hwt1d | 6.12 | 3.39 | 3.38 | 6.37 | 3.49 |![hwt1d](SVGs/hwt1d.svg) |
| hybridsort | 17.40 | 16.39 | 16.36 | 18.47 | 16.45 |![hybridsort](SVGs/hybridsort.svg) |
| hypterm | 10.92 | 15.09 | 15.05 | 18.92 | 20.58 |![hypterm](SVGs/hypterm.svg) |
| idivide | 7.92 | 6.23 | 6.23 | 11.15 | exe err |![idivide](SVGs/idivide.svg) |
| interleave | 6.02 | 3.46 | 3.47 | 5.30 | 3.39 |![interleave](SVGs/interleave.svg) |
| interval | 13.69 | 46.97 | 47.13 | 17.43 | 55.60 |![interval](SVGs/interval.svg) |
| intrinsics-cast | 69.54 | 9.21 | 9.22 | -- | -- |![intrinsics-cast](SVGs/intrinsics-cast.svg) |
| intrinsics-simd | 27.93 | -- | build err | -- | -- |![intrinsics-simd](SVGs/intrinsics-simd.svg) |
| inversek2j | 4.57 | 0.60 | 0.60 | 11.78 | 24.34 |![inversek2j](SVGs/inversek2j.svg) |
| is | 4.55 | 1.41 | 1.42 | -- | -- |![is](SVGs/is.svg) |
| ising | 33.95 | 15.69 | 15.71 | 33.13 | 22.85 |![ising](SVGs/ising.svg) |
| iso2dfd | 22.57 | 61.89 | 60.41 | 21.23 | 61.67 |![iso2dfd](SVGs/iso2dfd.svg) |
| jaccard | 197.46 | 254.66 | build err | -- | -- |![jaccard](SVGs/jaccard.svg) |
| jacobi | 4.81 | 8.91 | build err | 5.41 | 5.95 |![jacobi](SVGs/jacobi.svg) |
| jenkins-hash | 9.50 | 5.10 | 5.11 | 9.52 | 5.57 |![jenkins-hash](SVGs/jenkins-hash.svg) |
| kalman | 22.24 | 27.92 | 27.93 | 17.54 | 10.03 |![kalman](SVGs/kalman.svg) |
| keccaktreehash | 11.89 | 10.72 | 10.73 | 12.56 | 10.70 |![keccaktreehash](SVGs/keccaktreehash.svg) |
| keogh | 215.35 | 85.25 | 85.27 | 50.03 | 87.04 |![keogh](SVGs/keogh.svg) |
| kernelLaunch | 16.18 | 19.08 | 19.23 | 75.33 | 104.50 |![kernelLaunch](SVGs/kernelLaunch.svg) |
| kmc | 7.45 | 6.16 | 6.19 | -- | -- |![kmc](SVGs/kmc.svg) |
| kmeans | 32.53 | 42.64 | 42.41 | 30.35 | 49.75 |![kmeans](SVGs/kmeans.svg) |
| knn | 8.54 | 10.95 | 10.88 | 8.96 | 11.38 |![knn](SVGs/knn.svg) |
| kurtosis | 22.01 | 167.60 | 167.91 | -- | -- |![kurtosis](SVGs/kurtosis.svg) |
| lanczos | 6.90 | 3.34 | 3.26 | 7.31 | 3.91 |![lanczos](SVGs/lanczos.svg) |
| langevin | 73.86 | 10.60 | 12.36 | 74.80 | 16.99 |![langevin](SVGs/langevin.svg) |
| langford | 4.89 | 2.58 | 2.59 | build err | exe err |![langford](SVGs/langford.svg) |
| laplace | 5.84 | 4.45 | 4.45 | 9.40 | 21.52 |![laplace](SVGs/laplace.svg) |
| laplace3d | 95.07 | 18.61 | 17.75 | 56.92 | 22.79 |![laplace3d](SVGs/laplace3d.svg) |
| lavaMD | 104.12 | 61.88 | 61.77 | 194.21 | 55.93 |![lavaMD](SVGs/lavaMD.svg) |
| layernorm | 9.80 | 4.11 | build err | -- | -- |![layernorm](SVGs/layernorm.svg) |
| layout | 5.09 | 1.39 | 1.39 | 4.86 | 1.63 |![layout](SVGs/layout.svg) |
| lci | 19.97 | 8.41 | 8.41 | 4.37 | 7.82 |![lci](SVGs/lci.svg) |
| lda | 5.36 | 2.07 | 2.07 | build err | 14.90 |![lda](SVGs/lda.svg) |
| ldpc | 7.08 | 5.98 | 5.98 | 12.12 | 17.04 |![ldpc](SVGs/ldpc.svg) |
| lebesgue | 8.34 | 35.07 | | 6.62 | 36.22 |![lebesgue](SVGs/lebesgue.svg) |
| leukocyte | 4.76 | 1.16 | 1.02 | 6.07 | 1.15 |![leukocyte](SVGs/leukocyte.svg) |
| lfib4 | 188.11 | 31.34 | 28.44 | -- | -- |![lfib4](SVGs/lfib4.svg) |
| libor | 4.75 | 1.28 | 1.28 | 5.20 | 1.74 |![libor](SVGs/libor.svg) |
| lid-driven-cavity | 12.87 | 14.53 | 14.57 | 26.01 | 48.17 |![lid-driven-cavity](SVGs/lid-driven-cavity.svg) |
| lif | 133.75 | 76.25 | 76.29 | 150.82 | 80.63 |![lif](SVGs/lif.svg) |
| linearprobing | 106.78 | 76.77 | 76.54 | build err | exe err |![linearprobing](SVGs/linearprobing.svg) |
| log2 | 6.85 | 1.27 | 1.28 | exe err | 1.78 |![log2](SVGs/log2.svg) |
| logan | 10.75 | 15.29 | 15.64 | -- | -- |![logan](SVGs/logan.svg) |
| logic-resim | build err | 7.53 | 7.52 | -- | -- |![logic-resim](SVGs/logic-resim.svg) |
| logic-rewrite | 52.42 | build err | build err | -- | -- |![logic-rewrite](SVGs/logic-rewrite.svg) |
| logprob | 18.96 | 16.59 | build err | -- | -- |![logprob](SVGs/logprob.svg) |
| lombscargle | 4.83 | 1.18 | 1.18 | 4.88 | 1.18 |![lombscargle](SVGs/lombscargle.svg) |
| loopback | 7.82 | 10.90 | 10.88 | 10.54 | 11.34 |![loopback](SVGs/loopback.svg) |
| lr | 8.53 | 6.75 | 6.73 | 33.47 | 48.47 |![lr](SVGs/lr.svg) |
| lrn | 121.77 | 49.70 | 50.15 | 127.24 | 63.45 |![lrn](SVGs/lrn.svg) |
| lsqt | build err | 60.46 | build err | 59.87 | 74.43 |![lsqt](SVGs/lsqt.svg) |
| lud | 20.77 | 10.18 | 10.19 | 62.18 | 21.33 |![lud](SVGs/lud.svg) |
| ludb | 6.93 | 9.73 | build err | -- | -- |![ludb](SVGs/ludb.svg) |
| lulesh | 13.37 | 15.84 | 13.93 | 14.09 | 566.69 |![lulesh](SVGs/lulesh.svg) |
| lzss | | | | -- | -- |![lzss](SVGs/lzss.svg) |
| mallocFree | 5.60 | 9.40 | build err | 4.62 | 0.72 |![mallocFree](SVGs/mallocFree.svg) |
| mandelbrot | 7.37 | 16.41 | 15.85 | 5.63 | 13.34 |![mandelbrot](SVGs/mandelbrot.svg) |
| marchingCubes | 12.32 | 7.75 | build err | -- | -- |![marchingCubes](SVGs/marchingCubes.svg) |
| mask | 168.00 | 90.46 | 90.48 | 242.76 | 93.65 |![mask](SVGs/mask.svg) |
| match | 42.52 | 69.92 | 69.90 | 46.11 | 80.30 |![match](SVGs/match.svg) |
| matern | 17.87 | 31.70 | 31.69 | 116.98 | 111.40 |![matern](SVGs/matern.svg) |
| matrix-rotate | 30.57 | 9.78 | 10.18 | 31.16 | 10.20 |![matrix-rotate](SVGs/matrix-rotate.svg) |
| matrixT | 59.14 | 22.03 | 22.05 | -- | -- |![matrixT](SVGs/matrixT.svg) |
| maxFlops | 27.61 | 41.98 | 41.97 | 27.71 | 42.92 |![maxFlops](SVGs/maxFlops.svg) |
| maxpool3d | 40.06 | 15.72 | 14.82 | 40.38 | 15.40 |![maxpool3d](SVGs/maxpool3d.svg) |
| mcmd | 14.42 | 110.17 | 110.50 | 16.27 | 110.62 |![mcmd](SVGs/mcmd.svg) |
| mcpr | 12.75 | 60.69 | 60.66 | 14.44 | 67.98 |![mcpr](SVGs/mcpr.svg) |
| md | 18.53 | 19.47 | 19.51 | 18.97 | 18.84 |![md](SVGs/md.svg) |
| md5hash | 17.24 | 33.47 | 33.46 | 17.22 | 41.39 |![md5hash](SVGs/md5hash.svg) |
| mdh | 87.25 | 217.07 | build err | 40.38 | 218.96 |![mdh](SVGs/mdh.svg) |
| meanshift | 5.64 | 3.15 | 3.38 | 6.88 | 3.04 |![meanshift](SVGs/meanshift.svg) |
| medianfilter | 6.10 | 15.03 | 15.32 | 7.23 | 4.27 |![medianfilter](SVGs/medianfilter.svg) |
| memcpy | 7.45 | 19.46 | 19.69 | 8.45 | 18.21 |![memcpy](SVGs/memcpy.svg) |
| memtest | 20.17 | 28.04 | 28.04 | 21.57 | 38.18 |![memtest](SVGs/memtest.svg) |
| merge | 1657.99 | 1798.11 | 1799.55 | 1623.48 | 1814.24 |![merge](SVGs/merge.svg) |
| merkle | | 15.66 | build err | -- | -- |![merkle](SVGs/merkle.svg) |
| metropolis | 1511.38 | 71.73 | build err | | |![metropolis](SVGs/metropolis.svg) |
| mf-sgd | | | | -- | -- |![mf-sgd](SVGs/mf-sgd.svg) |
| michalewicz | 46.23 | 130.13 | 130.26 | | build err |![michalewicz](SVGs/michalewicz.svg) |
| miniDGS | | -- | | -- | -- |![miniDGS](SVGs/miniDGS.svg) |
| miniFE | | | | | |![miniFE](SVGs/miniFE.svg) |
| miniWeather | | build err | build err | | 127.07 |![miniWeather](SVGs/miniWeather.svg) |
| minibude | 5.27 | 2.47 | 2.47 | | build err |![minibude](SVGs/minibude.svg) |
| minimap2 | | build err | 1.87 | | build err |![minimap2](SVGs/minimap2.svg) |
| minimod | 91.44 | 2.85 | 2.86 | -- | -- |![minimod](SVGs/minimod.svg) |
| minisweep | 13.87 | 91.70 | 92.92 | | 2.97 |![minisweep](SVGs/minisweep.svg) |
| minkowski | 24.62 | 24.16 | 31.66 | | 89.66 |![minkowski](SVGs/minkowski.svg) |
| minmax | 273.04 | 4.51 | build err | -- | -- |![minmax](SVGs/minmax.svg) |
| mis | 94.24 | 0.47 | 0.47 | | |![mis](SVGs/mis.svg) |
| mixbench | 9.58 | build err | 50.78 | | build err |![mixbench](SVGs/mixbench.svg) |
| mmcsf | 8.26 | 8.10 | 8.14 | -- | -- |![mmcsf](SVGs/mmcsf.svg) |
| mnist | | 76.45 | 75.03 | -- | -- |![mnist](SVGs/mnist.svg) |
| morphology | 30.96 | 3.10 | 3.10 | | 0.92 |![morphology](SVGs/morphology.svg) |
| mpc | 4.55 | build err | build err | -- | -- |![mpc](SVGs/mpc.svg) |
| mr | 3.39 | 1.28 | 1.28 | | 2.39 |![mr](SVGs/mr.svg) |
| mrc | 117.28 | 4.17 | 4.12 | | build err |![mrc](SVGs/mrc.svg) |
| mrg32k3a | 241.19 | 0.84 | build err | -- | -- |![mrg32k3a](SVGs/mrg32k3a.svg) |
| mriQ | 6.97 | 6.90 | 28.14 | | build err |![mriQ](SVGs/mriQ.svg) |
| mt | | build err | build err | | build err |![mt](SVGs/mt.svg) |
| mtf | 66.29 | 20.05 | 20.00 | -- | -- |![mtf](SVGs/mtf.svg) |
| multimaterial | 30.05 | 28.64 | 28.69 | | build err |![multimaterial](SVGs/multimaterial.svg) |
| multinomial | 5.06 | 1.73 | 1.74 | -- | -- |![multinomial](SVGs/multinomial.svg) |
| murmurhash3 | 2.42 | 1.09 | 1.10 | | 1.87 |![murmurhash3](SVGs/murmurhash3.svg) |
| myocyte | 377.67 | 0.97 | 0.95 | | build err |![myocyte](SVGs/myocyte.svg) |
| nbnxm | | 8.56 | | -- | -- |![nbnxm](SVGs/nbnxm.svg) |
| nbody | | 0.49 | 0.49 | | 0.70 |![nbody](SVGs/nbody.svg) |
| ne | 8.15 | 2.30 | 2.27 | | build err |![ne](SVGs/ne.svg) |
| nlll | 48.67 | 149.47 | 149.47 | | build err |![nlll](SVGs/nlll.svg) |
| nms | | 0.61 | 0.60 | | build err |![nms](SVGs/nms.svg) |
| nn | 4.60 | 0.48 | 0.47 | | 0.73 |![nn](SVGs/nn.svg) |
| nonzero | 283.00 | 32.19 | 32.14 | -- | -- |![nonzero](SVGs/nonzero.svg) |
| norm2 | | build err | build err | | build err |![norm2](SVGs/norm2.svg) |
| nosync | exe err | 5.45 | build err | -- | -- |![nosync](SVGs/nosync.svg) |
| nqueen | 0.90 | 37.42 | 37.24 | | 28.63 |![nqueen](SVGs/nqueen.svg) |
| ntt | | 7.76 | 7.74 | | 10.11 |![ntt](SVGs/ntt.svg) |
| nw | 0.00 | 2.66 | 2.64 | | 2.96 |![nw](SVGs/nw.svg) |
| openmp | 16.40 | 66.84 | 66.87 | | 77.52 |![openmp](SVGs/openmp.svg) |
| opticalFlow | 5.92 | 3.05 | build err | -- | -- |![opticalFlow](SVGs/opticalFlow.svg) |
| overlap | | 0.56 | build err | -- | -- |![overlap](SVGs/overlap.svg) |
| overlay | 180.92 | 4.38 | 4.27 | | build err |![overlay](SVGs/overlay.svg) |
| p2p | | 0.78 | 0.81 | -- | -- |![p2p](SVGs/p2p.svg) |
| p4 | | 7.95 | 8.24 | | 3.29 |![p4](SVGs/p4.svg) |
| pad | 290.47 | 1.97 | build err | -- | -- |![pad](SVGs/pad.svg) |
| page-rank | 25.87 | 9.10 | 9.37 | | build err |![page-rank](SVGs/page-rank.svg) |
| particle-diffusion | | 9.68 | 9.66 | | 13.59 |![particle-diffusion](SVGs/particle-diffusion.svg) |
| particlefilter | 5.61 | 2.41 | 2.39 | | 4.97 |![particlefilter](SVGs/particlefilter.svg) |
| particles | | 1.99 | 1.99 | | build err |![particles](SVGs/particles.svg) |
| pathfinder | 54.42 | 1.35 | 1.35 | | 0.97 |![pathfinder](SVGs/pathfinder.svg) |
| pcc | 232.96 | build err | build err | -- | -- |![pcc](SVGs/pcc.svg) |
| perlin | | 13.10 | 13.06 | -- | -- |![perlin](SVGs/perlin.svg) |
| permutate | 27.69 | 34.46 | 34.64 | | 35.64 |![permutate](SVGs/permutate.svg) |
| permute | 120.73 | 1.06 | build err | | 1.78 |![permute](SVGs/permute.svg) |
| perplexity | 298.50 | 2.46 | 2.43 | | build err |![perplexity](SVGs/perplexity.svg) |
| phmm | | 93.71 | 93.71 | | 19.23 |![phmm](SVGs/phmm.svg) |
| pingpong | | build err | build err | -- | -- |![pingpong](SVGs/pingpong.svg) |
| pitch | | 4.11 | 4.11 | -- | -- |![pitch](SVGs/pitch.svg) |
| pnpoly | | 21.00 | 20.99 | | build err |![pnpoly](SVGs/pnpoly.svg) |
| pns | 34.52 | 7.48 | 7.48 | | 0.23 |![pns](SVGs/pns.svg) |
| pointwise | 92.62 | 1.96 | 3.06 | | build err |![pointwise](SVGs/pointwise.svg) |
| pool | 37.15 | 11.83 | 11.83 | | 21.57 |![pool](SVGs/pool.svg) |
| popcount | 939.39 | 4.31 | 4.30 | | 13.34 |![popcount](SVGs/popcount.svg) |
| prefetch | | 71.78 | 227.44 | -- | -- |![prefetch](SVGs/prefetch.svg) |
| present | | 4.64 | 4.64 | | 4.65 |![present](SVGs/present.svg) |
| prna | 0.00 | build err | build err | | |![prna](SVGs/prna.svg) |
| projectile | | 1.39 | 1.09 | | 4.29 |![projectile](SVGs/projectile.svg) |
| pso | 153.40 | 1.39 | 1.37 | | 1.92 |![pso](SVGs/pso.svg) |
| qem | | 11.34 | build err | -- | -- |![qem](SVGs/qem.svg) |
| qkv | | 13.80 | build err | -- | -- |![qkv](SVGs/qkv.svg) |
| qrg | | 20.09 | 20.10 | | build err |![qrg](SVGs/qrg.svg) |
| qtclustering | 2.96 | 0.94 | 0.94 | | build err |![qtclustering](SVGs/qtclustering.svg) |
| quicksort | | 12.10 | 12.03 | | build err |![quicksort](SVGs/quicksort.svg) |
| radixsort | | 1.66 | 1.67 | | build err |![radixsort](SVGs/radixsort.svg) |
| radixsort2 | 23.21 | 101.95 | 101.90 | -- | -- |![radixsort2](SVGs/radixsort2.svg) |
| rainflow | 42.57 | 7.21 | 7.21 | | 3.84 |![rainflow](SVGs/rainflow.svg) |
| randomAccess | | 14.06 | 13.91 | | 9.01 |![randomAccess](SVGs/randomAccess.svg) |
| rayleighBenardConvection | | 46.63 | build err | -- | -- |![rayleighBenardConvection](SVGs/rayleighBenardConvection.svg) |
| reaction | | 5.94 | 5.95 | | 22.70 |![reaction](SVGs/reaction.svg) |
| recursiveGaussian | | 4.35 | build err | | build err |![recursiveGaussian](SVGs/recursiveGaussian.svg) |
| relu | 1055.69 | 18.84 | build err | -- | -- |![relu](SVGs/relu.svg) |
| remap | | 22.12 | 21.89 | -- | -- |![remap](SVGs/remap.svg) |
| resize | 209.55 | 7.80 | 7.80 | | build err |![resize](SVGs/resize.svg) |
| resnet-kernels | 27.63 | 3.98 | 1.03 | -- | -- |![resnet-kernels](SVGs/resnet-kernels.svg) |
| reverse | | 1.72 | 1.71 | | build err |![reverse](SVGs/reverse.svg) |
| reverse2D | 35.86 | 1.45 | 2.66 | -- | -- |![reverse2D](SVGs/reverse2D.svg) |
| rfs | 358.55 | 13.29 | 13.14 | | build err |![rfs](SVGs/rfs.svg) |
| ring | 255.18 | 5.73 | 5.72 | -- | -- |![ring](SVGs/ring.svg) |
| rle | 173.36 | 0.68 | build err | -- | -- |![rle](SVGs/rle.svg) |
| rng-wallace | | 1.61 | 1.60 | | build err |![rng-wallace](SVGs/rng-wallace.svg) |
| rodrigues | 397.51 | 0.95 | 0.95 | | 5.43 |![rodrigues](SVGs/rodrigues.svg) |
| romberg | 91.07 | 1.13 | 1.10 | | build err |![romberg](SVGs/romberg.svg) |
| rotary | | 0.75 | build err | -- | -- |![rotary](SVGs/rotary.svg) |
| rowwiseMoments | | 2.43 | 2.43 | -- | -- |![rowwiseMoments](SVGs/rowwiseMoments.svg) |
| rsbench | | | | | 2.70 |![rsbench](SVGs/rsbench.svg) |
| rsc | 76.73 | 0.83 | 0.81 | | build err |![rsc](SVGs/rsc.svg) |
| rsmt | 5.39 | -- | | -- | -- |![rsmt](SVGs/rsmt.svg) |
| rtm8 | | 3.83 | 3.96 | | 4.58 |![rtm8](SVGs/rtm8.svg) |
| rushlarsen | 681.18 | 11.16 | 11.08 | | 11.25 |![rushlarsen](SVGs/rushlarsen.svg) |
| s3d | 2.93 | 0.64 | 0.62 | | 3.42 |![s3d](SVGs/s3d.svg) |
| s8n | 1236.81 | 28.46 | 28.40 | | 56.50 |![s8n](SVGs/s8n.svg) |
| sa | | 1.68 | 1.68 | -- | -- |![sa](SVGs/sa.svg) |
| sad | 4.78 | 2.08 | | | 8.00 |![sad](SVGs/sad.svg) |
| sampling | | 7.76 | 7.75 | | 2.15 |![sampling](SVGs/sampling.svg) |
| saxpy-ompt | | build err | build err | -- | -- |![saxpy-ompt](SVGs/saxpy-ompt.svg) |
| sc | | 1.14 | build err | -- | -- |![sc](SVGs/sc.svg) |
| scan | 1054.96 | 111.13 | 110.57 | | 104.33 |![scan](SVGs/scan.svg) |
| scan2 | 20.97 | 1.37 | 1.38 | | 4.09 |![scan2](SVGs/scan2.svg) |
| scan3 | 18.17 | 1.30 | 1.27 | -- | -- |![scan3](SVGs/scan3.svg) |
| scel | 22.06 | 12.28 | 23.28 | | build err |![scel](SVGs/scel.svg) |
| score | | 4.75 | build err | -- | -- |![score](SVGs/score.svg) |
| sddmm-batch | 206.93 | 212.57 | build err | -- | -- |![sddmm-batch](SVGs/sddmm-batch.svg) |
| seam-carving | 2.73 | 0.48 | | -- | -- |![seam-carving](SVGs/seam-carving.svg) |
| secp256k1 | | 1.28 | 1.28 | | 0.18 |![secp256k1](SVGs/secp256k1.svg) |
| segment-reduce | 560.80 | 8.57 | 8.60 | -- | -- |![segment-reduce](SVGs/segment-reduce.svg) |
| segsort | | 6.37 | build err | -- | -- |![segsort](SVGs/segsort.svg) |
| sheath | | 4.93 | 486.06 | | build err |![sheath](SVGs/sheath.svg) |
| shmembench | | 5.68 | 5.68 | | 1.11 |![shmembench](SVGs/shmembench.svg) |
| shuffle | | 7.95 | build err | -- | -- |![shuffle](SVGs/shuffle.svg) |
| si | | | | -- | -- |![si](SVGs/si.svg) |
| simpleMultiDevice | | 6.78 | 6.82 | -- | -- |![simpleMultiDevice](SVGs/simpleMultiDevice.svg) |
| simpleSpmv | 2123.72 | 13.45 | build err | | 10.00 |![simpleSpmv](SVGs/simpleSpmv.svg) |
| simplemoc | 213.07 | 4.38 | 4.30 | | 1261.04 |![simplemoc](SVGs/simplemoc.svg) |
| slit | | | | -- | -- |![slit](SVGs/slit.svg) |
| slu | | build err | | | build err |![slu](SVGs/slu.svg) |
| snake | 10.12 | 13.53 | 13.51 | 12.74 | 28.38 |![snake](SVGs/snake.svg) |
| sobel | 4.86 | 1.32 | 1.36 | 5.67 | 24.62 |![sobel](SVGs/sobel.svg) |
| sobol | | 3.27 | 3.27 | | 3.07 |![sobol](SVGs/sobol.svg) |
| softmax | | build err | build err | | 2.64 |![softmax](SVGs/softmax.svg) |
| softmax-fused | | 9.19 | build err | -- | -- |![softmax-fused](SVGs/softmax-fused.svg) |
| softmax-online | | 20.58 | build err | -- | -- |![softmax-online](SVGs/softmax-online.svg) |
| sort | | 6.06 | 6.05 | | build err |![sort](SVGs/sort.svg) |
| sortKV | | 39.88 | 39.58 | -- | -- |![sortKV](SVGs/sortKV.svg) |
| sosfil | | 6.63 | 6.62 | | build err |![sosfil](SVGs/sosfil.svg) |
| sparkler | | build err | build err | -- | -- |![sparkler](SVGs/sparkler.svg) |
| spaxpby | | build err | build err | -- | -- |![spaxpby](SVGs/spaxpby.svg) |
| spd2s | | 111.03 | build err | -- | -- |![spd2s](SVGs/spd2s.svg) |
| spgeam | | 19.13 | build err | -- | -- |![spgeam](SVGs/spgeam.svg) |
| spgemm | | 16.67 | build err | -- | -- |![spgemm](SVGs/spgemm.svg) |
| sph | | 3.06 | 3.06 | | build err |![sph](SVGs/sph.svg) |
| split | | 1.25 | 1.25 | | build err |![split](SVGs/split.svg) |
| spm | | 2.14 | 2.14 | | build err |![spm](SVGs/spm.svg) |
| spmm | | 7.19 | build err | -- | -- |![spmm](SVGs/spmm.svg) |
| spmv | | 6.13 | build err | -- | -- |![spmv](SVGs/spmv.svg) |
| spnnz | | 107.65 | build err | -- | -- |![spnnz](SVGs/spnnz.svg) |
| sps2d | | 118.14 | build err | -- | -- |![sps2d](SVGs/sps2d.svg) |
| spsm | | 117.88 | build err | -- | -- |![spsm](SVGs/spsm.svg) |
| spsort | | 104.18 | build err | -- | -- |![spsort](SVGs/spsort.svg) |
| sptrsv | 4.83 | 1.67 | 1.70 | | build err |![sptrsv](SVGs/sptrsv.svg) |
| srad | 4.51 | 0.54 | 0.54 | | build err |![srad](SVGs/srad.svg) |
| ss | 9.15 | 6.31 | 6.31 | | 922.90 |![ss](SVGs/ss.svg) |
| ssim | | 6.72 | build err | -- | -- |![ssim](SVGs/ssim.svg) |
| sss | 34.06 | build err | build err | -- | -- |![sss](SVGs/sss.svg) |
| sssp | 11.50 | 13.96 | 14.11 | -- | -- |![sssp](SVGs/sssp.svg) |
| stddev | | 24.36 | 33.47 | | build err |![stddev](SVGs/stddev.svg) |
| stencil1d | | 1.99 | 1.99 | | |![stencil1d](SVGs/stencil1d.svg) |
| stencil3d | | 6.76 | 6.70 | | 1.32 |![stencil3d](SVGs/stencil3d.svg) |
| streamCreateCopyDestroy | | 17.13 | 17.18 | -- | -- |![streamCreateCopyDestroy](SVGs/streamCreateCopyDestroy.svg) |
| streamOrderedAllocation | | 18.78 | 19.47 | -- | -- |![streamOrderedAllocation](SVGs/streamOrderedAllocation.svg) |
| streamPriority | | 2.84 | 1.44 | -- | -- |![streamPriority](SVGs/streamPriority.svg) |
| streamUM | | 24.85 | build err | -- | -- |![streamUM](SVGs/streamUM.svg) |
| streamcluster | | 4.05 | 4.03 | | build err |![streamcluster](SVGs/streamcluster.svg) |
| stsg | | build err | build err | -- | -- |![stsg](SVGs/stsg.svg) |
| su3 | | 2.28 | 2.28 | | build err |![su3](SVGs/su3.svg) |
| surfel | | 10.46 | 10.46 | | 179.57 |![surfel](SVGs/surfel.svg) |
| svd3x3 | 6.47 | 2.76 | 2.86 | 6.41 | 10.93 |![svd3x3](SVGs/svd3x3.svg) |
| sw4ck | | 25.07 | 24.92 | | 193.21 |![sw4ck](SVGs/sw4ck.svg) |
| swish | | 2.86 | 2.15 | | build err |![swish](SVGs/swish.svg) |
| tensorAccessor | | 17.51 | 17.51 | -- | -- |![tensorAccessor](SVGs/tensorAccessor.svg) |
| tensorT | | 1.81 | 1.82 | | 7.44 |![tensorT](SVGs/tensorT.svg) |
| testSNAP | 12.34 | 5.85 | 5.72 | 11.41 | 0.50 |![testSNAP](SVGs/testSNAP.svg) |
| thomas | | 17.48 | 17.45 | | 23.93 |![thomas](SVGs/thomas.svg) |
| threadfence | | 0.87 | 0.85 | | build err |![threadfence](SVGs/threadfence.svg) |
| tissue | | 21.07 | 21.06 | | build err |![tissue](SVGs/tissue.svg) |
| tonemapping | | 11.44 | 11.67 | | 3.98 |![tonemapping](SVGs/tonemapping.svg) |
| tpacf | 13.75 | 15.83 | build err | -- | -- |![tpacf](SVGs/tpacf.svg) |
| tqs | | 1.64 | 1.66 | | build err |![tqs](SVGs/tqs.svg) |
| triad | | 1.42 | 1.42 | | build err |![triad](SVGs/triad.svg) |
| tridiagonal | | 27.51 | 27.29 | | build err |![tridiagonal](SVGs/tridiagonal.svg) |
| tsa | | 1.95 | 1.78 | | build err |![tsa](SVGs/tsa.svg) |
| tsne | | build err | build err | -- | -- |![tsne](SVGs/tsne.svg) |
| tsp | 6.97 | 10.86 | 14.42 | | build err |![tsp](SVGs/tsp.svg) |
| unfold | | 0.62 | build err | -- | -- |![unfold](SVGs/unfold.svg) |
| urng | 4.68 | 0.45 | 0.45 | 4.57 | 0.79 |![urng](SVGs/urng.svg) |
| vanGenuchten | | 5.64 | 5.62 | | build err |![vanGenuchten](SVGs/vanGenuchten.svg) |
| vmc | | 1.85 | 1.85 | | build err |![vmc](SVGs/vmc.svg) |
| vol2col | | 10.06 | 10.03 | | build err |![vol2col](SVGs/vol2col.svg) |
| vote | | 12.21 | build err | -- | -- |![vote](SVGs/vote.svg) |
| voxelization | 12.44 | 94.03 | 94.03 | -- | -- |![voxelization](SVGs/voxelization.svg) |
| warpexchange | 5.47 | 0.66 | build err | -- | -- |![warpexchange](SVGs/warpexchange.svg) |
| warpsort | | 1.20 | build err | -- | -- |![warpsort](SVGs/warpsort.svg) |
| wedford | | 15.38 | build err | -- | -- |![wedford](SVGs/wedford.svg) |
| winograd | | 0.96 | 0.96 | | 3.51 |![winograd](SVGs/winograd.svg) |
| wlcpow | | 8.22 | 7.90 | | build err |![wlcpow](SVGs/wlcpow.svg) |
| wmma | 18.97 | 5.04 | build err | -- | -- |![wmma](SVGs/wmma.svg) |
| word2vec | 15.05 | build err | build err | -- | -- |![word2vec](SVGs/word2vec.svg) |
| wordcount | | 9.01 | 9.01 | 21.18 | 8.53 |![wordcount](SVGs/wordcount.svg) |
| wsm5 | | 9.75 | 9.73 | | build err |![wsm5](SVGs/wsm5.svg) |
| wyllie | | 3.43 | 3.42 | | 3.09 |![wyllie](SVGs/wyllie.svg) |
| xlqc | | build err | build err | | build err |![xlqc](SVGs/xlqc.svg) |
| xsbench | 51.69 | 55.05 | 43.60 | | 27.78 |![xsbench](SVGs/xsbench.svg) |
| zerocopy | | 114.24 | 988.62 | -- | -- |![zerocopy](SVGs/zerocopy.svg) |
| zeropoint | | 1.96 | 1.97 | | build err |![zeropoint](SVGs/zeropoint.svg) |
| zmddft | | 3.77 | 3.78 | | 3.10 |![zmddft](SVGs/zmddft.svg) |
| zoom | | 8.17 | 8.14 | -- | -- |![zoom](SVGs/zoom.svg) |
