#!/bin/sh
for n in *-omp
do
    echo $n
    n1=$n-aomp
    n2=$n-nvc
    cp -r $n $n1
    mv $n $n2
#    echo $n1
#    echo $n2
done
