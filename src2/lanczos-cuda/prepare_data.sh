#!/bin/sh
file1=../lanczos-cuda/data/social-large-800k.txt.bz2
#
file_out=../lanczos-cuda/data/social-large-800k.txt
#
if [ ! -e $file_out ]; then
    bunzip2 $file1
fi
