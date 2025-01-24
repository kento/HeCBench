#!/bin/bash

if [ -z "${1}" ];then
  echo Usage : ./gen_bench.sh BENCHMARK_NAME [skip cuda2hip yes or no]
  exit
else
  if [ -e ${1}-cuda ];then
    if [ "${2}" = "yes" ];then
      echo "skipping cuda2hip.sh"
    else
      ./scripts/cuda2hip.sh ${1}-cuda ${1}-hipified
    fi
    cd ${1}-hipified
    pwd
    ../scripts/edit_makefile.py -t AMD
    cd ..
    cd ${1}-cuda
    pwd
    ../scripts/edit_makefile.py -t NVIDIA --command
    cd ..
  fi
  if [ -e ${1}-hip ];then
    cd ${1}-hip
    pwd
    ../scripts/edit_makefile.py -t AMD --command
    cd ..
  fi
  if [ -e ${1}-omp ];then
    if [ -e ${1}-omp/Makefile.aomp ];then
      cp -R ${1}-omp ${1}-omp_aomp
      cd ${1}-omp_aomp
      pwd
      ../scripts/edit_makefile.py -t AMD -i Makefile.aomp
      cd ..
    fi
    if [ -e ${1}-omp/Makefile.nvc ];then
      cp -R ${1}-omp ${1}-omp_nvc
      cd ${1}-omp_nvc
      pwd
      ../scripts/edit_makefile.py -t NVIDIA -i Makefile.nvc
      cd ..
    fi
  else
    echo ${1}-omp does not exist
  fi
fi
