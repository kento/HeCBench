#!/bin/sh
for n in `cat List2`
do
    dir=$n-cuda
    pushd $dir
    cat Makefile.NVD | sed -e 's/sm_60/sm_90/g' | sed -e 's/sm_70/sm_90/g' > uuu2uuu
    mv uuu2uuu Makefile.NVD
    popd

    dir=$n-omp_nvc
    pushd $dir
    cat Makefile.NVD | sed -e 's/cc70/cc90/g'  > uuu2uuu
    mv uuu2uuu Makefile.NVD
    popd
done
