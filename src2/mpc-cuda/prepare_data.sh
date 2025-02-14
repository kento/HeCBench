#!/bin/sh
file1=../mpc-cuda/data/msg_sp.trace.fpc_part_aa
file2=../mpc-cuda/data/msg_sp.trace.fpc_part_ab
file3=../mpc-cuda/data/msg_sp.trace.fpc_part_ac
file4=../mpc-cuda/data/msg_sp.trace.fpc_part_ad
#
file_out=../mpc-cuda/msg_sp.trace.fpc
#
if [ ! -e $file_out ]; then
    cat $file1 $file2 $file3 $file4 > $file_out
fi
