#!/bin/sh
file1=../ge-spmm-cuda/data/snap/amazon0302/amazon0302.mtx.bz2
#
file_out=../ge-spmm-cuda/data/snap/amazon0302/amazon0302.mtx
#
if [ ! -e $file_out ]; then
    bunzip2 $file1
fi
