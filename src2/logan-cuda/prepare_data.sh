#!/bin/sh
file1=../logan-cuda/inputs/100k.txt.bz2_part_aa
file2=../logan-cuda/inputs/100k.txt.bz2_part_ab
file3=../logan-cuda/inputs/100k.txt.bz2_part_ac
file4=../logan-cuda/inputs/100k.txt.bz2_part_ad
#
fileM=../logan-cuda/inputs/100k.txt.bz2
#
file_out=../logan-cuda/inputs/100k.txt
#
if [ ! -e $file_out ]; then
    cat $file1 $file2 $file3 $file4 > $fileM
    bunzip2 $fileM
fi
