#!/bin/sh
file_out=../mpc-cuda/msg_sp.trace.fpc
#
if [ ! -e $file_out ]; then
    wget -P ../mpc-cuda --no-check-certificate http://www.cs.txstate.edu/~burtscher/research/datasets/FPdouble/msg_sp.trace.fpc
fi
