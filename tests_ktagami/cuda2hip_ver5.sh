#!/bin/sh
ORG=$1
NEW=$2
#
rm -rf $NEW
cp -r $ORG $NEW
#
cd $NEW
#
for n in `find . -print | grep "\.cu"`
do
    echo $n
    hipify-perl $n > utmptmp
    mv utmptmp $n
done
#
for n in `find . -print | grep "\.hpp"`
do
    echo $n
    hipify-perl $n > utmptmp
    mv utmptmp $n
done
#
#
for n in `find . -print | grep "\.c"`
do
    echo $n
    hipify-perl $n > utmptmp
    mv utmptmp $n
done
#
for n in `find . -print | grep "\.h"`
do
    echo $n
    hipify-perl $n > utmptmp
    mv utmptmp $n
done
#
for n in `find . -print | grep -i Makefile`
do
    echo $n
    cat $n | sed -e 's/nvcc/hipcc/g' | sed -e 's/-arch=\$(ARCH)//g' \
        | sed -e 's/-Xcompiler//g' \
        | sed -e 's/--use_fast_math//g' \
        | sed -e 's/-use_fast_math//g' \
        | sed -e 's/-maxrregcount=32//g' \
        | sed -e 's/cublas/hipblas/g' \
        | sed -e 's/--x cu/-x hip/g' \
       > utmptmp
    mv utmptmp $n
done
