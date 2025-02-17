#!/bin/sh
file1=../bmf-cuda/data/MNIST.in.bz2
#
file_out=../bmf-cuda/data/MNIST.in
#
if [ ! -e $file_out ]; then
    bunzip2 $file1
fi
