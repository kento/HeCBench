#!/bin/sh
for n in ../HeCBench_riken/src/*-cuda
do
    echo $n
    cp -r $n .
done
########
for n in ../HeCBench_riken/src/*-hip
do
    echo $n
    cp -r $n .
done
##########
for n in ../HeCBench_riken/src/*-omp
do
    echo $n
    cp -r $n .
done
##########
#for n in *-cuda
#do
#    echo $n
#    n1=`echo $n | sed -e 's/cuda/hipified/g'`
#    ./cuda2hip_ktagami5.sh $n $n1
#done
##########
for n in *-omp
do
    echo $n
    n1=$n.aomp
    n2=$n.nvc
    cp -r $n $n1
    mv $n $2
done
