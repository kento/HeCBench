#!/bin/sh
ORG=$1
NEW=$2
#
rm -rf $NEW
cp -r $ORG $NEW
#
cd $NEW
#
for n in `find . -print | grep ".cu"`
    do
        echo $n
        hipify-perl $n > utmptmp
        mv utmptmp $n
    done
#
for n in `find . -print | grep Makefile`
    do
        echo $n
        cat $n | sed -e 's/nvcc/hipcc/g' | sed -e 's/-arch=\$(ARCH)//g' \
        | sed -e 's/-Xcompiler//g' > utmptmp
        mv utmptmp $n
    done
