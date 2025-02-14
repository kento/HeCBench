#!/bin/sh
file1=../rsmt-cuda/data/newblue7.kraftwerk70.3d.80.20.82.m8.gr.gz
file_out=../rsmt-cuda/data/newblue7.kraftwerk70.3d.80.20.82.m8.gr
#
if [ ! -e $file_out ]; then
    gunzip $file1
fi

