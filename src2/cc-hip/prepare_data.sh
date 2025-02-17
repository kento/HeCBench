#!/bin/sh
file_out=../cc-cuda/delaunay_n24.egr
#
export WGET_HOME=../lib/wget2-2.2.0/
#
if [ ! -e $file_out ]; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WGET_HOME/lib
    $WGET_HOME/bin/wget2 https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/delaunay_n24.egr -P ../cc-cuda  1> /dev/null
fi
