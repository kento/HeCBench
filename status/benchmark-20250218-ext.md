| name | cuda | hip | hipified | omp_nvc | omp_aomp | plot |
| -- | -- | -- | -- | -- | -- | -- |
| accuracy | 5.55 | 6.49 | 6.48 | 63.77 | 10.19 |![accuracy](SVGs/accuracy.svg) |
| ace | 9.52 | 3.35 | 3.32 | 5.15 | exe err |![ace](SVGs/ace.svg) |
| adam | 7.08 | 4.20 | 4.19 | 8.65 | exe err |![adam](SVGs/adam.svg) |
| addBiasResidualLayerNorm | 18.92 | 50.36 | exe err | -- | -- |![addBiasResidualLayerNorm](SVGs/addBiasResidualLayerNorm.svg) |
| adv | 7.32 | 12.87 | 13.06 | 7.81 | 4.22 |![adv](SVGs/adv.svg) |
| aes | 8.95 | 1.23 | 1.21 | 8.99 | 0.74 |![aes](SVGs/aes.svg) |
| affine | 7.76 | 4.91 | 4.72 | 14.42 | 32.31 |![affine](SVGs/affine.svg) |
| aidw | 11.02 | 9.75 | 48.61 | 11.46 | 87.89 |![aidw](SVGs/aidw.svg) |
| aligned-types | 5.70 | 2.16 | 2.16 | 6.91 | 3.42 |![aligned-types](SVGs/aligned-types.svg) |
| all-pairs-distance | 6.19 | 22.04 | 21.94 | 20.97 | 3.41 |![all-pairs-distance](SVGs/all-pairs-distance.svg) |
| allreduce | 67.40 | 44.81 | 44.71 | -- | -- |![allreduce](SVGs/allreduce.svg) |
| amgmk | 4.53 | 0.75 | 0.74 | 4.52 | 0.80 |![amgmk](SVGs/amgmk.svg) |
| ans | 6.89 | 3.15 | 3.29 | exe err | 3.53 |![ans](SVGs/ans.svg) |
| aobench | 4.48 | 0.48 | 0.48 | 7.60 | 1.53 |![aobench](SVGs/aobench.svg) |
| aop | 8.77 | 34.60 | 39.66 | 306.19 | exe err |![aop](SVGs/aop.svg) |
| asmooth | 16.35 | 12.53 | 13.39 | 9.84 | 17.43 |![asmooth](SVGs/asmooth.svg) |
| assert | 4.36 | exe err | exe err | -- | -- |![assert](SVGs/assert.svg) |
| asta | 6.24 | 3.67 | 3.69 | 12.83 | exe err |![asta](SVGs/asta.svg) |
| atan2 | 4.68 | 0.75 | 0.76 | 4.59 | 0.77 |![atan2](SVGs/atan2.svg) |
| atomicAggregate | 7.63 | 5.51 | build err | -- | -- |![atomicAggregate](SVGs/atomicAggregate.svg) |
| atomicCAS | 17.72 | 14.55 | 15.02 | -- | -- |![atomicCAS](SVGs/atomicCAS.svg) |
| atomicCost | 97.45 | 7.86 | 7.93 | 96.88 | 25.15 |![atomicCost](SVGs/atomicCost.svg) |
| atomicIntrinsics | 89.73 | 70.29 | 70.31 | 4.58 | 0.90 |![atomicIntrinsics](SVGs/atomicIntrinsics.svg) |
| atomicPerf | 11.11 | 5.23 | 9153.82 | 12.14 | 367.77 |![atomicPerf](SVGs/atomicPerf.svg) |
| atomicReduction | 5.09 | 0.92 | 0.92 | 4.94 | 1.79 |![atomicReduction](SVGs/atomicReduction.svg) |
| atomicSystemWide | 5.31 | 10.19 | 11.58 | -- | -- |![atomicSystemWide](SVGs/atomicSystemWide.svg) |
| attention | 4.93 | 19.14 | 19.41 | 86.14 | 30.19 |![attention](SVGs/attention.svg) |
| attentionMultiHead | 5.36 | 1.66 | build err | -- | -- |![attentionMultiHead](SVGs/attentionMultiHead.svg) |
| axhelm | 4.45 | 0.79 | 0.79 | 4.52 | 0.81 |![axhelm](SVGs/axhelm.svg) |
| b+tree | 4.63 | 0.92 | 0.92 | 4.56 | 0.87 |![b+tree](SVGs/b+tree.svg) |
| babelstream | 4.64 | 0.95 | 0.95 | 4.65 | 3.11 |![babelstream](SVGs/babelstream.svg) |
| background-subtract | 5.31 | 4.41 | 4.72 | 7.17 | 4.49 |![background-subtract](SVGs/background-subtract.svg) |
| backprop | 4.51 | 0.46 | 0.46 | 4.94 | 0.25 |![backprop](SVGs/backprop.svg) |
| bezier-surface | 18.36 | 23.78 | 23.78 | 17.16 | 23.87 |![bezier-surface](SVGs/bezier-surface.svg) |
| bfs | 5.01 | 1.15 | 1.10 | 4.84 | 1.07 |![bfs](SVGs/bfs.svg) |
| bh | 8.16 | 6715.84 | 9574.28 | -- | -- |![bh](SVGs/bh.svg) |
| bicgstab | 1.02 | build err | over 120 | -- | -- |![bicgstab](SVGs/bicgstab.svg) |
| bilateral | 9.52 | 46.55 | 18.20 | 10.11 | 548.56 |![bilateral](SVGs/bilateral.svg) |
| bincount | 5.48 | 2.01 | 2.01 | -- | -- |![bincount](SVGs/bincount.svg) |
| binomial | 4.92 | 1.78 | 1.78 | -- | 1.65 |![binomial](SVGs/binomial.svg) |
| bitcracker | 17.68 | 24.08 | 24.11 | -- | -- |![bitcracker](SVGs/bitcracker.svg) |
| bitonic-sort | 16.45 | 13.34 | 13.31 | 11.35 | 14.64 |![bitonic-sort](SVGs/bitonic-sort.svg) |
| bitpacking | 8.14 | 4.10 | 4.09 | -- | -- |![bitpacking](SVGs/bitpacking.svg) |
| bitpermute | 6.01 | 7.06 | build err | -- | -- |![bitpermute](SVGs/bitpermute.svg) |
| black-scholes | 9.55 | 3.97 | 3.95 | 7.78 | 8.32 |![black-scholes](SVGs/black-scholes.svg) |
| blas-dot | 8.90 | 10.00 | 10.01 | -- | -- |![blas-dot](SVGs/blas-dot.svg) |
| blas-fp8gemm | 4.10 | 5.63 | build err | -- | -- |![blas-fp8gemm](SVGs/blas-fp8gemm.svg) |
| blas-gemm | 12.53 | 4.48 | 4.48 | -- | -- |![blas-gemm](SVGs/blas-gemm.svg) |
| blas-gemmBatched | 19.85 | 21.57 | 21.59 | -- | -- |![blas-gemmBatched](SVGs/blas-gemmBatched.svg) |
| blas-gemmEx | 15.44 | over 300 | build err | -- | -- |![blas-gemmEx](SVGs/blas-gemmEx.svg) |
| blas-gemmEx2 | 15.35 | over 300 | build err | -- | -- |![blas-gemmEx2](SVGs/blas-gemmEx2.svg) |
| blas-gemmStridedBatched | 17.48 | 21.57 | build err | -- | -- |![blas-gemmStridedBatched](SVGs/blas-gemmStridedBatched.svg) |
| blockAccess | 5.76 | 8.15 | 8.13 | -- | -- |![blockAccess](SVGs/blockAccess.svg) |
| blockexchange | 33.67 | 0.67 | 0.67 | -- | -- |![blockexchange](SVGs/blockexchange.svg) |
| bm3d | 5.32 | 6.80 | 6.71 | -- | -- |![bm3d](SVGs/bm3d.svg) |
| bmf | exe err | 7.23 | build err | -- | -- |![bmf](SVGs/bmf.svg) |
| bn | 6.33 | exe err | exe err | exe err | 33.49 |![bn](SVGs/bn.svg) |
| bonds | 14.44 | 19.26 | 19.28 | 16.35 | 15.18 |![bonds](SVGs/bonds.svg) |
| boxfilter | 8.75 | 9.16 | 9.16 | 11.78 | 37.13 |![boxfilter](SVGs/boxfilter.svg) |
| bscan | 9.54 | 1.78 | 0.44 | -- | -- |![bscan](SVGs/bscan.svg) |
| bsearch | 743.47 | 4.61 | 4.61 | exe err | 34.57 |![bsearch](SVGs/bsearch.svg) |
| bspline-vgh | 14.11 | 10.08 | 10.08 | 9.98 | 11.30 |![bspline-vgh](SVGs/bspline-vgh.svg) |
| bsw | 4.21 | 0.53 | build err | -- | -- |![bsw](SVGs/bsw.svg) |
| btree | 5.43 | -- | build err | -- | -- |![btree](SVGs/btree.svg) |
| burger | 110.83 | 21.37 | 21.44 | 75.83 | 25.65 |![burger](SVGs/burger.svg) |
| bwt | 209.63 | 13.38 | 13.66 | exe err | 15.18 |![bwt](SVGs/bwt.svg) |
| car | 11.59 | 19.95 | 19.95 | 11.59 | 49.90 |![car](SVGs/car.svg) |
| cbsfil | 43.47 | 5.93 | 5.94 | 42.67 | 8.36 |![cbsfil](SVGs/cbsfil.svg) |
| cc | 5.11 | 2.49 | build err | -- | -- |![cc](SVGs/cc.svg) |
| ccl | 42.14 | build err | build err | -- | -- |![ccl](SVGs/ccl.svg) |
| ccs | 5.36 | over 300 | over 300 | 5.36 | over 300 |![ccs](SVGs/ccs.svg) |
| ccsd-trpdrv | 41.53 | 19.17 | 19.35 | 35.93 | 2.34 |![ccsd-trpdrv](SVGs/ccsd-trpdrv.svg) |
| ced | 6.51 | 2.76 | 2.74 | -- | -- |![ced](SVGs/ced.svg) |
| cfd | 4.67 | 1.46 | 1.48 | 4.93 | 2.65 |![cfd](SVGs/cfd.svg) |
| chacha20 | 6.78 | 8.25 | 8.24 | 8.89 | 7.34 |![chacha20](SVGs/chacha20.svg) |
| channelShuffle | 49.94 | 38.92 | 32.81 | 68.93 | 76.38 |![channelShuffle](SVGs/channelShuffle.svg) |
| channelSum | 66.15 | 52.23 | 52.13 | 69.99 | 295.96 |![channelSum](SVGs/channelSum.svg) |
| che | 5.18 | 2.69 | 2.69 | 5.25 | 14.97 |![che](SVGs/che.svg) |
| chemv | 5.08 | 1.90 | 1.90 | 5.54 | build err |![chemv](SVGs/chemv.svg) |
| chi2 | 223.36 | 20.42 | 20.40 | 260.91 | 184.59 |![chi2](SVGs/chi2.svg) |
| clenergy | 4.83 | 1.34 | 1.34 | 5.26 | 17.58 |![clenergy](SVGs/clenergy.svg) |
| clink | 5.71 | 5.67 | 5.12 | 7.66 | 6.56 |![clink](SVGs/clink.svg) |
| clock | 4.48 | 0.44 | 0.42 | -- | -- |![clock](SVGs/clock.svg) |
| cm | 1352.85 | 4307.41 | 4303.46 | build err | build err |![cm](SVGs/cm.svg) |
| cmembench | 6.34 | 10.19 | 10.18 | -- | -- |![cmembench](SVGs/cmembench.svg) |
| cmp | 8.14 | 12.45 | 12.46 | 9.27 | 14.24 |![cmp](SVGs/cmp.svg) |
| cobahh | 480.09 | 684.64 | 672.18 | 417.25 | 657.95 |![cobahh](SVGs/cobahh.svg) |
| collision | 9.90 | 2.58 | build err | -- | -- |![collision](SVGs/collision.svg) |
| colorwheel | 48.47 | 2.35 | 2.35 | 42.53 | 3.46 |![colorwheel](SVGs/colorwheel.svg) |
| columnarSolver | 11.32 | exe err | exe err | 11.60 | exe err |![columnarSolver](SVGs/columnarSolver.svg) |
| complex | 25.25 | 3.68 | 3.67 | 15.22 | 26.06 |![complex](SVGs/complex.svg) |
| compute-score | 6.84 | 4.74 | 4.74 | 8.43 | 1.92 |![compute-score](SVGs/compute-score.svg) |
| concat | 26.60 | 13.27 | 13.17 | 26.21 | 57.74 |![concat](SVGs/concat.svg) |
| concurrentKernels | 283.38 | 31.63 | 31.63 | -- | -- |![concurrentKernels](SVGs/concurrentKernels.svg) |
| contract | 1497.61 | 15.19 | 15.24 | 1418.37 | 24.22 |![contract](SVGs/contract.svg) |
| conversion | 6.77 | 0.91 | 0.91 | 5.45 | 1.49 |![conversion](SVGs/conversion.svg) |
| convolution1D | 1715.16 | 100.98 | 100.98 | 3870.95 | 193.78 |![convolution1D](SVGs/convolution1D.svg) |
| convolution3D | 2.86 | 75.10 | build err | 4.63 | 190.53 |![convolution3D](SVGs/convolution3D.svg) |
| convolutionDeformable | build err | 10.12 | 10.32 | -- | -- |![convolutionDeformable](SVGs/convolutionDeformable.svg) |
| convolutionSeparable | 19.86 | 26.99 | 25.70 | 30.44 | 25.65 |![convolutionSeparable](SVGs/convolutionSeparable.svg) |
| cooling | 2269.87 | 1.05 | 1.85 | 1115.12 | 8.40 |![cooling](SVGs/cooling.svg) |
| coordinates | 510.74 | 3.35 | 3.34 | -- | -- |![coordinates](SVGs/coordinates.svg) |
| copy | 30.88 | 15.66 | 2659.07 | -- | -- |![copy](SVGs/copy.svg) |
| crc64 | 159.62 | 4.06 | 4.05 | 168.38 | 4.52 |![crc64](SVGs/crc64.svg) |
| cross | 10.92 | 11.47 | 11.47 | 681.38 | 10.64 |![cross](SVGs/cross.svg) |
| crossEntropy | 11.71 | 10.41 | 10.40 | -- | -- |![crossEntropy](SVGs/crossEntropy.svg) |
| crs | 1259.70 | 10.30 | 10.30 | | 11.64 |![crs](SVGs/crs.svg) |
| d2q9-bgk | 4.59 | 1.46 | 1.47 | 6.04 | 5.30 |![d2q9-bgk](SVGs/d2q9-bgk.svg) |
| d3q19-bgk | 8.73 | 24.91 | 24.99 | -- | -- |![d3q19-bgk](SVGs/d3q19-bgk.svg) |
| damage | 25.55 | 1.02 | 1.02 | 222.52 | 0.79 |![damage](SVGs/damage.svg) |
| daphne | 2.50 | 0.78 | 0.79 | -- | -- |![daphne](SVGs/daphne.svg) |
| dct8x8 | 181.81 | 1.50 | 1.50 | exe err | 1.25 |![dct8x8](SVGs/dct8x8.svg) |
| ddbp | 18.57 | 4.88 | 4.88 | 18.44 | 9.96 |![ddbp](SVGs/ddbp.svg) |
| debayer | 15.96 | 0.93 | 1.08 | exe err | 1.39 |![debayer](SVGs/debayer.svg) |
| degrid | 17.95 | 48.95 | 51.49 | 22.38 | 113.63 |![degrid](SVGs/degrid.svg) |
| dense-embedding | 198.45 | 180.11 | 180.26 | 1777.93 | 721.85 |![dense-embedding](SVGs/dense-embedding.svg) |
| depixel | 1163.31 | 30.85 | 30.91 | 6489.09 | 12.29 |![depixel](SVGs/depixel.svg) |
| deredundancy | 18.25 | 59.04 | 59.03 | 18.18 | 43.62 |![deredundancy](SVGs/deredundancy.svg) |
| determinant | 4.73 | 0.67 | 0.67 | -- | -- |![determinant](SVGs/determinant.svg) |
| diamond | 40.77 | 45.36 | 46.11 | 40.97 | build err |![diamond](SVGs/diamond.svg) |
| dispatch | 6.37 | 1.98 | 2.02 | -- | -- |![dispatch](SVGs/dispatch.svg) |
| distort | 4.51 | 0.52 | 0.51 | 4.57 | 1.52 |![distort](SVGs/distort.svg) |
| divergence | 6.50 | 1.64 | 1.64 | 5.03 | 4.81 |![divergence](SVGs/divergence.svg) |
| doh | 18.82 | 19.78 | 19.77 | 6.18 | 19.63 |![doh](SVGs/doh.svg) |
| dp | 54.96 | 15.64 | 15.62 | 34.07 | 31.72 |![dp](SVGs/dp.svg) |
| dpid | 50.59 | 1.01 | build err | -- | -- |![dpid](SVGs/dpid.svg) |
| dropout | 31.01 | 0.93 | 0.93 | -- | -- |![dropout](SVGs/dropout.svg) |
| dslash | 6.75 | 2.19 | 4.42 | 15.43 | 4.68 |![dslash](SVGs/dslash.svg) |
| dwconv | 81.64 | 23.33 | 23.22 | -- | -- |![dwconv](SVGs/dwconv.svg) |
| dwconv1d | 6.80 | 19.33 | 19.45 | -- | -- |![dwconv1d](SVGs/dwconv1d.svg) |
| dxtc1 | 4.19 | 0.61 | 0.63 | -- | -- |![dxtc1](SVGs/dxtc1.svg) |
| dxtc2 | 4.23 | 0.46 | 0.46 | 4.37 | 0.41 |![dxtc2](SVGs/dxtc2.svg) |
| easyWave | 7.11 | 2.48 | 2.46 | 7.25 | 3.17 |![easyWave](SVGs/easyWave.svg) |
| ecdh | 37.30 | 2.97 | 2.98 | 40.94 | 43.45 |![ecdh](SVGs/ecdh.svg) |
| egs | 4.96 | exe err | build err | -- | -- |![egs](SVGs/egs.svg) |
| eigenvalue | 23.90 | 13.44 | 13.40 | 22.87 | 16.98 |![eigenvalue](SVGs/eigenvalue.svg) |
| eikonal | 206.17 | 10.50 | 9.93 | -- | -- |![eikonal](SVGs/eikonal.svg) |
| entropy | 57.12 | 10.30 | 10.29 | 57.16 | 19.05 |![entropy](SVGs/entropy.svg) |
| epistasis | 67.92 | 189.79 | 189.71 | 136.80 | 710.62 |![epistasis](SVGs/epistasis.svg) |
| ert | 7.38 | 3.53 | 3.53 | -- | -- |![ert](SVGs/ert.svg) |
| expdist | 9.41 | 7.21 | 7.21 | 15.58 | 5.28 |![expdist](SVGs/expdist.svg) |
| extend2 | 7.56 | 11.37 | 11.34 | 6.81 | 0.79 |![extend2](SVGs/extend2.svg) |
| extrema | 8.83 | 6.38 | 6.40 | 9.40 | 20.70 |![extrema](SVGs/extrema.svg) |
| f16max | 25.83 | 35.51 | 35.49 | -- | -- |![f16max](SVGs/f16max.svg) |
| f16sp | 33.71 | 30.80 | 1868.33 | -- | -- |![f16sp](SVGs/f16sp.svg) |
| face | 4.68 | 1.05 | 1.05 | build err | exe err |![face](SVGs/face.svg) |
| fdtd3d | 35.55 | 4.92 | 4.67 | 49.94 | 4.43 |![fdtd3d](SVGs/fdtd3d.svg) |
| feynman-kac | 84.36 | 114.00 | 115.15 | 24.12 | 46.84 |![feynman-kac](SVGs/feynman-kac.svg) |
| fft | 6.52 | 2.18 | 2.18 | 7.49 | 0.49 |![fft](SVGs/fft.svg) |
| fhd | 765.79 | 1.11 | 1.09 | | 32.30 |![fhd](SVGs/fhd.svg) |
| filter | 174.29 | 15.03 | 15.04 | 174.81 | 7.50 |![filter](SVGs/filter.svg) |
| flame | 5.83 | 3.90 | 3.93 | -- | -- |![flame](SVGs/flame.svg) |
| flip | 85.49 | 10.28 | 10.30 | 90.87 | 43.39 |![flip](SVGs/flip.svg) |
| floydwarshall | 78.88 | 1.92 | 1.95 | 97.80 | 5.67 |![floydwarshall](SVGs/floydwarshall.svg) |
| floydwarshall2 | 10.25 | 6.18 | 6.38 | -- | -- |![floydwarshall2](SVGs/floydwarshall2.svg) |
| fluidSim | 22.99 | 25.92 | 25.54 | 21.98 | 30.36 |![fluidSim](SVGs/fluidSim.svg) |
| fpc | 70.35 | 1.40 | 1.40 | exe err | 0.80 |![fpc](SVGs/fpc.svg) |
| fpdc | 6.45 | 2.98 | 2.96 | build err | exe err |![fpdc](SVGs/fpdc.svg) |
| frechet | 15.26 | 1.11 | 1.11 | exe err | 1.04 |![frechet](SVGs/frechet.svg) |
| fresnel | 8.66 | 6.65 | build err | build err | 13.98 |![fresnel](SVGs/fresnel.svg) |
| frna | build err | 346.24 | 347.30 | exe err | over 360 |![frna](SVGs/frna.svg) |
| fsm | 5.16 | 6.99 | 7.00 | build err | 0.34 |![fsm](SVGs/fsm.svg) |
| fwt | 5.36 | 1.60 | 1.59 | 5.83 | 3.69 |![fwt](SVGs/fwt.svg) |
| ga | 52.84 | 5.01 | 5.02 | 36.55 | 5.93 |![ga](SVGs/ga.svg) |
| gabor | 120.39 | 4.92 | 7.98 | exe err | 51.57 |![gabor](SVGs/gabor.svg) |
| gamma-correction | 2178.54 | 19.22 | 19.21 | 3350.52 | 16.87 |![gamma-correction](SVGs/gamma-correction.svg) |
| gaussian | 20.74 | 5.63 | 5.66 | 9.21 | 9.71 |![gaussian](SVGs/gaussian.svg) |
| gc | 4.24 | exe err | build err | build err | exe err |![gc](SVGs/gc.svg) |
| gd | 12.87 | 12.91 | 28.89 | 13.65 | 28.57 |![gd](SVGs/gd.svg) |
| ge-spmm | 7.26 | 8.56 | 8.55 | -- | -- |![ge-spmm](SVGs/ge-spmm.svg) |
| geam | 127.40 | 20.72 | 20.72 | -- | -- |![geam](SVGs/geam.svg) |
| gels | 4.67 | 1.83 | build err | -- | -- |![gels](SVGs/gels.svg) |
| gelu | 57.17 | 58.40 | 57.65 | -- | -- |![gelu](SVGs/gelu.svg) |
| gemv | 9.81 | 78.18 | build err | -- | -- |![gemv](SVGs/gemv.svg) |
| geodesic | 6.55 | 3.83 | 3.82 | 6.42 | 4.02 |![geodesic](SVGs/geodesic.svg) |
| gerbil | exe err | 3.06 | build err | -- | -- |![gerbil](SVGs/gerbil.svg) |
| gibbs | 4.53 | 0.71 | 0.70 | -- | -- |![gibbs](SVGs/gibbs.svg) |
| glu | 156.41 | exe err | 118.35 | 1593.95 | 164.83 |![glu](SVGs/glu.svg) |
| gmm | 299.98 | 1.92 | 1.03 | 341.15 | 3.68 |![gmm](SVGs/gmm.svg) |
| goulash | 142.59 | 11.60 | 11.66 | 147.07 | 20.07 |![goulash](SVGs/goulash.svg) |
| gpp | 6.83 | 238.43 | 234.16 | 6.99 | 29.99 |![gpp](SVGs/gpp.svg) |
| graphB+ | 7.21 | over 3600 | build err | -- | -- |![graphB+](SVGs/graphB+.svg) |
| graphExecution | 6.60 | 1.84 | 1.84 | -- | -- |![graphExecution](SVGs/graphExecution.svg) |
| grep | 923.81 | 96.57 | 96.60 | 473.83 | 30.55 |![grep](SVGs/grep.svg) |
| grrt | 17.00 | 55.86 | 52.29 | 15.40 | |![grrt](SVGs/grrt.svg) |
| gru | 26.44 | 48.42 | 48.44 | -- | -- |![gru](SVGs/gru.svg) |
| haccmk | 6.54 | 4.25 | 4.25 | 6.38 | 5.41 |![haccmk](SVGs/haccmk.svg) |
| halo-finder | 5.44 | 4.50 | 7.77 | -- | -- |![halo-finder](SVGs/halo-finder.svg) |
| hausdorff | 32.73 | 14.40 | 14.40 | 33.92 | 15.64 |![hausdorff](SVGs/hausdorff.svg) |
| haversine | 6.46 | 1.82 | 2.05 | 6.52 | 2.09 |![haversine](SVGs/haversine.svg) |
| hbc | 15.68 | 15.25 | 15.24 | -- | -- |![hbc](SVGs/hbc.svg) |
| heartwall | 4.66 | 1.97 | 1.93 | exe err | exe err |![heartwall](SVGs/heartwall.svg) |
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
| kalman | 22.24 | 27.92 | 27.93 | exe err | 10.03 |![kalman](SVGs/kalman.svg) |
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
| lebesgue | 8.34 | 35.07 | 18.50 | 6.62 | 36.22 |![lebesgue](SVGs/lebesgue.svg) |
| leukocyte | 4.76 | 1.16 | 1.02 | 6.07 | 1.15 |![leukocyte](SVGs/leukocyte.svg) |
| lfib4 | 188.11 | 31.34 | 28.44 | -- | -- |![lfib4](SVGs/lfib4.svg) |
| libor | 4.75 | 1.28 | 1.28 | 5.20 | 1.74 |![libor](SVGs/libor.svg) |
| lid-driven-cavity | 12.87 | 14.53 | 14.57 | 26.01 | 48.17 |![lid-driven-cavity](SVGs/lid-driven-cavity.svg) |
| lif | 133.75 | 76.25 | 76.29 | 150.82 | 80.63 |![lif](SVGs/lif.svg) |
| linearprobing | 106.78 | 76.77 | 76.54 | build err | exe err |![linearprobing](SVGs/linearprobing.svg) |
| log2 | 6.85 | 1.27 | 1.28 | exe err | 1.78 |![log2](SVGs/log2.svg) |
| logan | 10.75 | 15.29 | 15.64 | -- | -- |![logan](SVGs/logan.svg) |
| logic-resim | 8.50 | 7.53 | 7.52 | -- | -- |![logic-resim](SVGs/logic-resim.svg) |
| logic-rewrite | 52.42 | build err | build err | -- | -- |![logic-rewrite](SVGs/logic-rewrite.svg) |
| logprob | 18.96 | 16.59 | build err | -- | -- |![logprob](SVGs/logprob.svg) |
| lombscargle | 4.83 | 1.18 | 1.18 | 4.88 | 1.18 |![lombscargle](SVGs/lombscargle.svg) |
| loopback | 7.82 | 10.90 | 10.88 | 10.54 | 11.34 |![loopback](SVGs/loopback.svg) |
| lr | 8.53 | 6.75 | 6.73 | 33.47 | 48.47 |![lr](SVGs/lr.svg) |
| lrn | 121.77 | 49.70 | 50.15 | 127.24 | 63.45 |![lrn](SVGs/lrn.svg) |
| lsqt | 22.07 | 60.46 | build err | 36.89 | 74.43 |![lsqt](SVGs/lsqt.svg) |
| lud | 20.77 | 10.18 | 10.19 | 62.18 | 21.33 |![lud](SVGs/lud.svg) |
| ludb | 6.93 | 9.73 | build err | -- | -- |![ludb](SVGs/ludb.svg) |
| lulesh | 13.37 | 15.84 | 13.93 | 14.09 | 566.69 |![lulesh](SVGs/lulesh.svg) |
| lzss | 0.00 | | | -- | -- |![lzss](SVGs/lzss.svg) |
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
| merkle | 8.14 | 15.19 | 15.58 | -- | -- |![merkle](SVGs/merkle.svg) |
| metropolis | 1511.71 | 71.73 | build err | TLE error | TLE error |![metropolis](SVGs/metropolis.svg) |
| mf-sgd | build err | 0.33 | 0.33 | -- | -- |![mf-sgd](SVGs/mf-sgd.svg) |
| michalewicz | 48.89 | 130.13 | 130.26 | TLE error | 138.14 |![michalewicz](SVGs/michalewicz.svg) |
| miniDGS | exe err | -- | exe err | -- | -- |![miniDGS](SVGs/miniDGS.svg) |
| miniFE | 9.03 | 10.13 | 10.35 | 10.11 | 10.31 |![miniFE](SVGs/miniFE.svg) |
| miniWeather | 5.82 | 3.15 | 3.12 | 6.34 | 64.82 |![miniWeather](SVGs/miniWeather.svg) |
| minibude | 3.33 | 2.44 | 2.46 | 5.55 | 3.06 |![minibude](SVGs/minibude.svg) |
| minimap2 | 5.23 | build err | 1.76 | 4.98 | 13.85 |![minimap2](SVGs/minimap2.svg) |
| minimod | 5.82 | 2.85 | 2.86 | -- | -- |![minimod](SVGs/minimod.svg) |
| minisweep | 80.75 | 91.70 | 92.92 | exe err | 2.97 |![minisweep](SVGs/minisweep.svg) |
| minkowski | 25.36 | 24.16 | 31.66 | 23.81 | 89.66 |![minkowski](SVGs/minkowski.svg) |
| minmax | 221.71 | 231.56 | 230.54 | -- | -- |![minmax](SVGs/minmax.svg) |
| mis | 95.76 | 0.47 | 0.47 | exe err | TLE error |![mis](SVGs/mis.svg) |
| mixbench | 9.17 | 50.64 | 50.64 | 9.88 | 53.22 |![mixbench](SVGs/mixbench.svg) |
| mmcsf | 7.62 | 8.10 | 8.14 | -- | -- |![mmcsf](SVGs/mmcsf.svg) |
| mnist | 20.00 | 70.23 | 71.51 | -- | -- |![mnist](SVGs/mnist.svg) |
| morphology | 26.72 | 3.10 | 3.10 | TLE error | 0.92 |![morphology](SVGs/morphology.svg) |
| mpc | 8.55 | 0.38 | exe err | -- | -- |![mpc](SVGs/mpc.svg) |
| mr | 4.85 | 1.28 | 1.28 | exe err | 2.39 |![mr](SVGs/mr.svg) |
| mrc | 85.53 | 4.17 | 4.12 | TLE error | 5.53 |![mrc](SVGs/mrc.svg) |
| mrg32k3a | 209.77 | 128.15 | 128.06 | -- | -- |![mrg32k3a](SVGs/mrg32k3a.svg) |
| mriQ | 4.34 | 6.77 | 27.77 | 5.10 | 30.97 |![mriQ](SVGs/mriQ.svg) |
| mt | 5.24 | 4.61 | 4.63 | 4.44 | 4.26 |![mt](SVGs/mt.svg) |
| mtf | 12.27 | 20.05 | 20.00 | -- | -- |![mtf](SVGs/mtf.svg) |
| multimaterial | 28.21 | 16.72 | 16.87 | 25.84 | 83.99 |![multimaterial](SVGs/multimaterial.svg) |
| multinomial | 6.26 | 1.73 | 1.74 | -- | -- |![multinomial](SVGs/multinomial.svg) |
| murmurhash3 | 2.72 | 2.87 | 2.89 | 4.52 | 23.13 |![murmurhash3](SVGs/murmurhash3.svg) |
| myocyte | 34.40 | 69.84 | 87.86 | 12.58 | 13.34 |![myocyte](SVGs/myocyte.svg) |
| nbnxm | 4.78 | 8.56 | TLE error | -- | -- |![nbnxm](SVGs/nbnxm.svg) |
| nbody | 46.57 | 0.49 | 0.49 | 64.85 | 0.70 |![nbody](SVGs/nbody.svg) |
| ne | 4.86 | 3.17 | 3.18 | 4.58 | 3.79 |![ne](SVGs/ne.svg) |
| nlll | 52.01 | 149.47 | 149.47 | TLE error | 146.98 |![nlll](SVGs/nlll.svg) |
| nms | 4.37 | 0.61 | 0.60 | exe err | 1.31 |![nms](SVGs/nms.svg) |
| nn | 4.28 | 0.48 | 0.47 | 4.19 | 0.73 |![nn](SVGs/nn.svg) |
| nonzero | 49.52 | 32.19 | 32.14 | -- | -- |![nonzero](SVGs/nonzero.svg) |
| norm2 | 3.67 | 3.51 | 3.39 | 4.98 | 5.07 |![norm2](SVGs/norm2.svg) |
| nosync | 9.43 | 5.41 | 7.49 | -- | -- |![nosync](SVGs/nosync.svg) |
| nqueen | 7.87 | 37.42 | 37.24 | 8.55 | 28.63 |![nqueen](SVGs/nqueen.svg) |
| ntt | 8.20 | 7.76 | 7.74 | 8.87 | 10.11 |![ntt](SVGs/ntt.svg) |
| nw | 10.16 | 2.66 | 2.64 | 23.34 | 2.96 |![nw](SVGs/nw.svg) |
| openmp | 15.40 | 66.84 | 66.87 | 15.35 | 77.52 |![openmp](SVGs/openmp.svg) |
| opticalFlow | 5.66 | 3.03 | exe err | -- | -- |![opticalFlow](SVGs/opticalFlow.svg) |
| overlap | 4.19 | 0.56 | build err | -- | -- |![overlap](SVGs/overlap.svg) |
| overlay | 5.80 | 4.28 | 4.28 | 10.60 | 20.40 |![overlay](SVGs/overlay.svg) |
| p2p | 3.92 | 0.77 | 0.76 | -- | -- |![p2p](SVGs/p2p.svg) |
| p4 | 5.39 | 7.95 | 8.24 | 627.37 | 3.29 |![p4](SVGs/p4.svg) |
| pad | 292.22 | 1.97 | build err | -- | -- |![pad](SVGs/pad.svg) |
| page-rank | 25.46 | 25.72 | 25.69 | 25.80 | 24.79 |![page-rank](SVGs/page-rank.svg) |
| particle-diffusion | 12.92 | 9.68 | 9.66 | 12.89 | 13.59 |![particle-diffusion](SVGs/particle-diffusion.svg) |
| particlefilter | 6.10 | 2.41 | 2.39 | exe err | 4.97 |![particlefilter](SVGs/particlefilter.svg) |
| particles | 5.35 | 1.99 | 1.99 | TLE error | exe err |![particles](SVGs/particles.svg) |
| pathfinder | 53.00 | 1.35 | 1.35 | exe err | 0.97 |![pathfinder](SVGs/pathfinder.svg) |
| pcc | 5.21 | 8.05 | 8.06 | -- | -- |![pcc](SVGs/pcc.svg) |
| perlin | 10.29 | 13.10 | 13.06 | -- | -- |![perlin](SVGs/perlin.svg) |
| permutate | 28.86 | 34.37 | 34.49 | 28.04 | 34.79 |![permutate](SVGs/permutate.svg) |
| permute | 40.01 | 1.03 | 1.03 | 41.56 | 2.20 |![permute](SVGs/permute.svg) |
| perplexity | 279.74 | 2.46 | 2.43 | 214.51 | 2.73 |![perplexity](SVGs/perplexity.svg) |
| phmm | 5.37 | 93.71 | 93.71 | 9.49 | 19.23 |![phmm](SVGs/phmm.svg) |
| pingpong | 252.47 | build err | build err | -- | -- |![pingpong](SVGs/pingpong.svg) |
| pitch | 5.35 | 4.11 | 4.11 | -- | -- |![pitch](SVGs/pitch.svg) |
| pnpoly | 6.94 | 20.91 | 20.91 | 118.97 | 226.78 |![pnpoly](SVGs/pnpoly.svg) |
| pns | 6.69 | 7.48 | 7.48 | exe err | 0.23 |![pns](SVGs/pns.svg) |
| pointwise | 25.07 | 1.96 | 3.06 | 171.66 | exe err |![pointwise](SVGs/pointwise.svg) |
| pool | 30.62 | 11.83 | 11.83 | 39.38 | 21.57 |![pool](SVGs/pool.svg) |
| popcount | 838.56 | 4.31 | 4.30 | 175.36 | 13.34 |![popcount](SVGs/popcount.svg) |
| prefetch | 6.99 | 71.78 | 227.44 | -- | -- |![prefetch](SVGs/prefetch.svg) |
| present | 6.93 | 4.64 | 4.64 | 6.99 | 4.65 |![present](SVGs/present.svg) |
| prna | 81.94 | 512.73 | 518.59 | exe err | 556.08 |![prna](SVGs/prna.svg) |
| projectile | 4.75 | 1.39 | 1.09 | 4.69 | 4.29 |![projectile](SVGs/projectile.svg) |
| pso | 4.71 | 1.39 | 1.37 | 4.45 | 1.92 |![pso](SVGs/pso.svg) |
| qem | 7.77 | 11.50 | 11.90 | -- | -- |![qem](SVGs/qem.svg) |
| qkv | 19.18 | 13.98 | 13.93 | -- | -- |![qkv](SVGs/qkv.svg) |
| qrg | 7.15 | 20.03 | 20.03 | 10.33 | 30.61 |![qrg](SVGs/qrg.svg) |
| qtclustering | 38.31 | 0.94 | 0.94 | exe err | exe err |![qtclustering](SVGs/qtclustering.svg) |
| quicksort | 40.20 | 51.31 | 51.09 | exe err | 49.43 |![quicksort](SVGs/quicksort.svg) |
| radixsort | 2.91 | 1.69 | 1.68 | 8.22 | 5.00 |![radixsort](SVGs/radixsort.svg) |
| radixsort2 | 8.56 | 101.95 | 101.90 | -- | -- |![radixsort2](SVGs/radixsort2.svg) |
| rainflow | 42.50 | 7.21 | 7.21 | 38.72 | 3.84 |![rainflow](SVGs/rainflow.svg) |
| randomAccess | 9.99 | 14.06 | 13.91 | 26.06 | 9.01 |![randomAccess](SVGs/randomAccess.svg) |
| rayleighBenardConvection | 32.25 | 42.94 | 46.43 | -- | -- |![rayleighBenardConvection](SVGs/rayleighBenardConvection.svg) |
| reaction | 1.88 | 5.94 | 5.95 | 9.53 | 22.70 |![reaction](SVGs/reaction.svg) |
| recursiveGaussian | 5.60 | 4.37 | 4.38 | exe err | exe err |![recursiveGaussian](SVGs/recursiveGaussian.svg) |
| relu | 1026.76 | 18.84 | build err | -- | -- |![relu](SVGs/relu.svg) |
| remap | 46.57 | 22.12 | 21.89 | -- | -- |![remap](SVGs/remap.svg) |
| resize | 208.50 | 7.80 | 7.80 | exe err | 8.67 |![resize](SVGs/resize.svg) |
| resnet-kernels | 26.28 | 3.98 | 1.03 | -- | -- |![resnet-kernels](SVGs/resnet-kernels.svg) |
| reverse | 3.39 | 1.61 | 1.70 | 7.93 | 10.46 |![reverse](SVGs/reverse.svg) |
| reverse2D | 33.68 | 1.45 | 2.66 | -- | -- |![reverse2D](SVGs/reverse2D.svg) |
| rfs | 357.77 | 13.29 | 13.14 | exe err | 12.78 |![rfs](SVGs/rfs.svg) |
| ring | 9.13 | 5.73 | 5.72 | -- | -- |![ring](SVGs/ring.svg) |
| rle | 108.12 | 0.68 | build err | -- | -- |![rle](SVGs/rle.svg) |
| rng-wallace | 4.78 | 1.61 | 1.60 | 5.70 | 1.77 |![rng-wallace](SVGs/rng-wallace.svg) |
| rodrigues | 400.05 | 0.95 | 0.95 | 299.68 | 5.43 |![rodrigues](SVGs/rodrigues.svg) |
| romberg | 2.68 | 1.11 | 1.10 | 4.57 | 0.94 |![romberg](SVGs/romberg.svg) |
| rotary | 2.43 | 0.75 | 0.73 | -- | -- |![rotary](SVGs/rotary.svg) |
| rowwiseMoments | 203.27 | 2.43 | 2.43 | -- | -- |![rowwiseMoments](SVGs/rowwiseMoments.svg) |
| rsbench | 4.66 | 3.73 | 3.74 | exe err | 2.71 |![rsbench](SVGs/rsbench.svg) |
| rsc | 4.29 | 0.85 | 0.90 | 4.38 | 0.85 |![rsc](SVGs/rsc.svg) |
| rsmt | 9.03 | -- | build err | -- | -- |![rsmt](SVGs/rsmt.svg) |
| rtm8 | 6.17 | 3.83 | 3.96 | 6.15 | 4.58 |![rtm8](SVGs/rtm8.svg) |
| rushlarsen | 684.29 | 11.16 | 11.08 | 688.86 | 11.25 |![rushlarsen](SVGs/rushlarsen.svg) |
| s3d | 6.53 | 0.64 | 0.62 | 9.14 | 3.42 |![s3d](SVGs/s3d.svg) |
| s8n | 1206.32 | 28.46 | 28.40 | 1268.07 | 56.50 |![s8n](SVGs/s8n.svg) |
| sa | 83.02 | 2.50 | 2.49 | -- | -- |![sa](SVGs/sa.svg) |
| sad | 4.61 | 2.08 | 2.08 | over 300 | 2.10 |![sad](SVGs/sad.svg) |
| sampling | 7.49 | 7.76 | 7.75 | 8.85 | 2.15 |![sampling](SVGs/sampling.svg) |
| saxpy-ompt | exe err | 53.86 | 53.08 | -- | -- |![saxpy-ompt](SVGs/saxpy-ompt.svg) |
| sc | TLE error | 1.14 | build err | -- | -- |![sc](SVGs/sc.svg) |
| scan | 620.44 | 111.13 | 110.57 | 762.38 | 104.33 |![scan](SVGs/scan.svg) |
| scan2 | 5.11 | 1.37 | 1.38 | 8.98 | 4.09 |![scan2](SVGs/scan2.svg) |
| scan3 | 5.17 | 1.30 | 1.27 | -- | -- |![scan3](SVGs/scan3.svg) |
| scel | 16.97 | 46.08 | 89.70 | 724.30 | 47.40 |![scel](SVGs/scel.svg) |
| score | 12.04 | 4.73 | exe err | -- | -- |![score](SVGs/score.svg) |
| sddmm-batch | 207.90 | 213.08 | 212.23 | -- | -- |![sddmm-batch](SVGs/sddmm-batch.svg) |
| seam-carving | 4.22 | 0.48 | 0.48 | -- | -- |![seam-carving](SVGs/seam-carving.svg) |
| secp256k1 | 4.62 | 1.28 | 1.28 | 4.42 | 0.18 |![secp256k1](SVGs/secp256k1.svg) |
| segment-reduce | 149.02 | 8.57 | 8.60 | -- | -- |![segment-reduce](SVGs/segment-reduce.svg) |
| segsort | 9.35 | 6.37 | build err | -- | -- |![segsort](SVGs/segsort.svg) |
| sheath | 7.26 | 4.93 | 486.06 | 7.54 | 327.06 |![sheath](SVGs/sheath.svg) |
| shmembench | 6.89 | 5.68 | 5.68 | exe err | 1.11 |![shmembench](SVGs/shmembench.svg) |
| shuffle | 8.89 | 7.95 | build err | -- | -- |![shuffle](SVGs/shuffle.svg) |
| si | 8.51 | 1.21 | build err | -- | -- |![si](SVGs/si.svg) |
| simpleMultiDevice | 5.39 | 6.78 | 6.82 | -- | -- |![simpleMultiDevice](SVGs/simpleMultiDevice.svg) |
| simpleSpmv | 2291.54 | 13.45 | build err | 2110.89 | 10.00 |![simpleSpmv](SVGs/simpleSpmv.svg) |
| simplemoc | 232.48 | 4.38 | 4.30 | 1234.66 | 1261.04 |![simplemoc](SVGs/simplemoc.svg) |
| slit | 4.33 | build err | build err | -- | -- |![slit](SVGs/slit.svg) |
| slu | build err | 20.95 | 20.94 | build err | 27.24 |![slu](SVGs/slu.svg) |
| snake | 9.84 | 13.48 | 13.49 | 12.36 | 16.13 |![snake](SVGs/snake.svg) |
| sobel | 4.69 | 1.29 | 1.35 | 5.29 | 6.36 |![sobel](SVGs/sobel.svg) |
| sobol | 6.27 | 3.27 | 3.27 | 6.32 | 3.07 |![sobol](SVGs/sobol.svg) |
| softmax | 43.51 | 29.39 | build err | 45.71 | 116.69 |![softmax](SVGs/softmax.svg) |
| softmax-fused | 4.55 | 9.19 | build err | -- | -- |![softmax-fused](SVGs/softmax-fused.svg) |
| softmax-online | 30.85 | 20.58 | build err | -- | -- |![softmax-online](SVGs/softmax-online.svg) |
| sort | 7.01 | 6.06 | 6.05 | 39.25 | exe err |![sort](SVGs/sort.svg) |
| sortKV | 363.11 | 39.88 | 39.58 | -- | -- |![sortKV](SVGs/sortKV.svg) |
| sosfil | 5.99 | 6.63 | 6.62 | 15.29 | 10.23 |![sosfil](SVGs/sosfil.svg) |
| sparkler | 6.27 | 7.14 | 7.13 | -- | -- |![sparkler](SVGs/sparkler.svg) |
| spaxpby | 1172.55 | 64.45 | 65.22 | -- | -- |![spaxpby](SVGs/spaxpby.svg) |
| spd2s | 2649.11 | 114.44 | 118.63 | -- | -- |![spd2s](SVGs/spd2s.svg) |
| spgeam | 71.22 | 19.41 | 19.45 | -- | -- |![spgeam](SVGs/spgeam.svg) |
| spgemm | 1046.42 | 16.67 | 16.51 | -- | -- |![spgemm](SVGs/spgemm.svg) |
| sph | 7.19 | 3.06 | 3.06 | 7.28 | 3.76 |![sph](SVGs/sph.svg) |
| split | 50.64 | 1.25 | 1.25 | 187.37 | exe err |![split](SVGs/split.svg) |
| spm | 181.14 | 2.14 | 2.14 | 177.96 | 3.33 |![spm](SVGs/spm.svg) |
| spmm | 12.19 | 7.10 | 7.16 | -- | -- |![spmm](SVGs/spmm.svg) |
| spmv | 2523.57 | 6.16 | 6.13 | -- | -- |![spmv](SVGs/spmv.svg) |
| spnnz | 407.44 | 107.15 | 106.60 | -- | -- |![spnnz](SVGs/spnnz.svg) |
| sps2d | 1636.62 | 118.07 | 118.92 | -- | -- |![sps2d](SVGs/sps2d.svg) |
| spsm | 123.23 | 117.88 | build err | -- | -- |![spsm](SVGs/spsm.svg) |
| spsort | 333.51 | 103.84 | 104.25 | -- | -- |![spsort](SVGs/spsort.svg) |
| sptrsv | 4.73 | 1.50 | 1.70 | 4.82 | 2.12 |![sptrsv](SVGs/sptrsv.svg) |
| srad | 62.96 | 54.97 | 54.89 | 183.00 | 64.05 |![srad](SVGs/srad.svg) |
| ss | 9.03 | 6.31 | 6.31 | 19.89 | 922.90 |![ss](SVGs/ss.svg) |
| ssim | 6.04 | 6.66 | 6.65 | -- | -- |![ssim](SVGs/ssim.svg) |
| sss | 34.11 | 7.16 | 7.15 | -- | -- |![sss](SVGs/sss.svg) |
| sssp | 11.10 | 13.02 | 13.20 | -- | -- |![sssp](SVGs/sssp.svg) |
| stddev | 49.52 | 24.36 | 33.47 | 52.98 | 38.77 |![stddev](SVGs/stddev.svg) |
| stencil1d | 5.44 | 2.62 | 2.66 | 59.09 | 37.35 |![stencil1d](SVGs/stencil1d.svg) |
| stencil3d | 20.34 | 6.76 | 6.70 | exe err | 1.32 |![stencil3d](SVGs/stencil3d.svg) |
| streamCreateCopyDestroy | 5.74 | 17.13 | 17.18 | -- | -- |![streamCreateCopyDestroy](SVGs/streamCreateCopyDestroy.svg) |
| streamOrderedAllocation | 6.60 | 18.78 | 19.47 | -- | -- |![streamOrderedAllocation](SVGs/streamOrderedAllocation.svg) |
| streamPriority | 4.65 | 2.84 | 1.44 | -- | -- |![streamPriority](SVGs/streamPriority.svg) |
| streamUM | 43.61 | 32.24 | 28.33 | -- | -- |![streamUM](SVGs/streamUM.svg) |
| streamcluster | 20.70 | 26.78 | 31.44 | build err | exe err |![streamcluster](SVGs/streamcluster.svg) |
| stsg | 25.65 | 208.03 | 207.85 | -- | -- |![stsg](SVGs/stsg.svg) |
| su3 | 9.93 | 2.28 | 2.28 | 25.95 | 3.02 |![su3](SVGs/su3.svg) |
| surfel | 2178.24 | 10.46 | 10.46 | 2425.75 | 179.57 |![surfel](SVGs/surfel.svg) |
| svd3x3 | 5.90 | 2.71 | 2.82 | 5.92 | 2.71 |![svd3x3](SVGs/svd3x3.svg) |
| sw4ck | 6.43 | 25.07 | 24.92 | exe err | 193.21 |![sw4ck](SVGs/sw4ck.svg) |
| swish | 26.70 | 2.86 | 2.15 | 1427.21 | 2.34 |![swish](SVGs/swish.svg) |
| tensorAccessor | 60.39 | 17.51 | 17.51 | -- | -- |![tensorAccessor](SVGs/tensorAccessor.svg) |
| tensorT | 4.64 | 1.81 | 1.82 | 4.84 | 7.44 |![tensorT](SVGs/tensorT.svg) |
| testSNAP | 5.77 | 5.85 | 5.72 | 6.07 | 0.50 |![testSNAP](SVGs/testSNAP.svg) |
| thomas | 1573.13 | 17.48 | 17.45 | 1390.69 | 23.93 |![thomas](SVGs/thomas.svg) |
| threadfence | 34.79 | 0.87 | 0.85 | 37.17 | 0.50 |![threadfence](SVGs/threadfence.svg) |
| tissue | 13.77 | 21.07 | 21.06 | 15.19 | 19.99 |![tissue](SVGs/tissue.svg) |
| tonemapping | 5.15 | 10.71 | 10.71 | 4.92 | 11.03 |![tonemapping](SVGs/tonemapping.svg) |
| tpacf | 12.33 | 15.83 | build err | -- | -- |![tpacf](SVGs/tpacf.svg) |
| tqs | 4.38 | 1.64 | 1.66 | exe err | exe err |![tqs](SVGs/tqs.svg) |
| triad | 4.45 | 1.42 | 1.42 | 4.53 | 2.86 |![triad](SVGs/triad.svg) |
| tridiagonal | 91.46 | 27.51 | 27.29 | 161.09 | 36.25 |![tridiagonal](SVGs/tridiagonal.svg) |
| tsa | 1266.41 | 1.95 | 1.78 | 1484.54 | 1.82 |![tsa](SVGs/tsa.svg) |
| tsne | build err | build err | build err | -- | -- |![tsne](SVGs/tsne.svg) |
| tsp | 6.72 | 10.86 | 14.42 | build err | 10.82 |![tsp](SVGs/tsp.svg) |
| unfold | 39.14 | 0.59 | 0.60 | -- | -- |![unfold](SVGs/unfold.svg) |
| urng | 4.21 | 0.45 | 0.45 | 4.36 | 0.79 |![urng](SVGs/urng.svg) |
| vanGenuchten | 31.55 | 5.64 | 5.62 | 44.96 | 6.03 |![vanGenuchten](SVGs/vanGenuchten.svg) |
| vmc | 4.57 | 1.80 | 1.81 | exe err | 1.69 |![vmc](SVGs/vmc.svg) |
| vol2col | 8.24 | 10.06 | 10.03 | 7.95 | exe err |![vol2col](SVGs/vol2col.svg) |
| vote | 5.56 | 12.21 | build err | -- | -- |![vote](SVGs/vote.svg) |
| voxelization | 12.13 | 94.01 | 94.05 | -- | -- |![voxelization](SVGs/voxelization.svg) |
| warpexchange | 51.19 | 0.67 | 0.66 | -- | -- |![warpexchange](SVGs/warpexchange.svg) |
| warpsort | 5.79 | 1.20 | build err | -- | -- |![warpsort](SVGs/warpsort.svg) |
| wedford | 198.58 | 15.38 | build err | -- | -- |![wedford](SVGs/wedford.svg) |
| winograd | 4.70 | 0.96 | 0.96 | 4.82 | 3.51 |![winograd](SVGs/winograd.svg) |
| wlcpow | 6.25 | 8.22 | 7.90 | 16.99 | 8.79 |![wlcpow](SVGs/wlcpow.svg) |
| wmma | 103.29 | 5.04 | build err | -- | -- |![wmma](SVGs/wmma.svg) |
| word2vec | 13.15 | 23.95 | 23.96 | -- | -- |![word2vec](SVGs/word2vec.svg) |
| wordcount | 21.50 | 9.01 | 9.01 | 20.92 | 8.53 |![wordcount](SVGs/wordcount.svg) |
| wsm5 | 7.28 | 9.75 | 9.73 | 9.99 | 10.72 |![wsm5](SVGs/wsm5.svg) |
| wyllie | 755.63 | 468.16 | 440.62 | over 800 | 441.99 |![wyllie](SVGs/wyllie.svg) |
| xlqc | 4.22 | 0.48 | 0.47 | 4.14 | 0.48 |![xlqc](SVGs/xlqc.svg) |
| xsbench | 48.31 | 55.05 | 43.60 | 6.98 | 27.78 |![xsbench](SVGs/xsbench.svg) |
| zerocopy | 21.70 | 114.24 | 988.62 | -- | -- |![zerocopy](SVGs/zerocopy.svg) |
| zeropoint | 18.85 | 151.86 | 151.68 | 1443.93 | 152.91 |![zeropoint](SVGs/zeropoint.svg) |
| zmddft | 4.46 | 3.77 | 3.78 | 18.94 | 3.10 |![zmddft](SVGs/zmddft.svg) |
| zoom | 40.79 | 8.17 | 8.14 | -- | -- |![zoom](SVGs/zoom.svg) |
