#!/bin/sh
for n in `cat List2`
do
    dir=$n-hip
    pushd $dir
    cat Makefile.AMD | sed -e 's/sm_60/sm_90/g' | sed -e 's/sm_70/sm_90/g' > uuu2uuu
    mv uuu2uuu Makefile.AMD
    popd

    dir=$n-hipified
    pushd $dir
    cat Makefile.AMD | sed -e 's/sm_60/sm_90/g' | sed -e 's/sm_70/sm_90/g' > uuu2uuu
    mv uuu2uuu Makefile.AMD
    popd

    dir=$n-omp_aomp
    pushd $dir
    cat Makefile.AMD | sed -e 's/gfx906/gfx90a/g' | sed -e 's/gfx907/gfx90a/g' | sed -e 's/gfx908/gfx90a/g' > uuu2uuu
    mv uuu2uuu Makefile.AMD
    popd
done
