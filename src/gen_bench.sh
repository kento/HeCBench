#!/bin/bash

if [ -z "${1}" ];then
  echo Usage : ./gen_bench.sh BENCHMARK_NAME
  exit
else
  if [ -e ${1}-cuda ];then
    cd ${1}-cuda
    pwd
    ../edit_makefile.py -t NVIDIA
    cd ..
    ./cuda2hip.sh ${1}-cuda ${1}-hipified
    cd ${1}-hipified
    pwd
    ../edit_makefile.py -t AMD
    cd ..
  fi
  if [ -e ${1}-hip ];then
    cd ${1}-hip
    pwd
    ../edit_makefile.py -t AMD
    cd ..
  fi
  if [ -e ${1}-omp ];then
    if [ -e ${1}-omp/Makefile.aomp ];then
      cp -R ${1}-omp ${1}-omp_aomp
      cd ${1}-omp_aomp
      pwd
      ../edit_makefile.py -t AMD -i Makefile.aomp
      cd ..
    fi
    if [ -e ${1}-omp/Makefile.nvc ];then
      cp -R ${1}-omp ${1}-omp_nvc
      cd ${1}-omp_nvc
      pwd
      ../edit_makefile.py -t NVIDIA -i Makefile.nvc
      cd ..
    fi
  else
    echo ${1}-omp does not exist
  fi
fi
