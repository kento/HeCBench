#!/bin/sh
file_out=../cc-cuda/delaunay_n24.egr
#
if [ ! -e $file_out ]; then
    wget https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/delaunay_n24.egr -P ../cc-cuda
fi
