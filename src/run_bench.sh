#!/bin/bash

if [ -z "${1}" ];then
  echo Usage : ./run_bench.sh BENCHMARK_NAME GPU_TYPE
  exit
else
  echo "running ${1} on ${2}"
  echo "command "
  cat ${1}-${2}-cmd
  if [[ "${2}" == "NVIDIA" ]];then
    if [ -e ${1}-cuda ];then
      echo "running benchmark under ${1}-cuda"
      cd ${1}-cuda
      make -f Makefile.NVD run_time > log 2>&1
      cd ..
    fi
    if [ -e ${1}-omp_nvc ];then
      echo "running benchmark under ${1}-omp_nvc"
      cd ${1}-omp_nvc
      make -f Makefile.NVD run_time > log 2>&1
      cd ..
    fi
  elif [[ "${2}" == "AMD" ]];then
    echo "running benchmark under ${1}-hip"
    if [ -e ${1}-hip ];then
      cd ${1}-hip
      make -f Makefile.AMD run_time > log 2>&1
      cd ..
    fi
    if [ -e ${1}-hipified ];then
      echo "running benchmark under ${1}-hipified"
      cd ${1}-hipified
      make -f Makefile.AMD run_time > log 2>&1
      cd ..
    fi
    if [ -e ${1}-omp_aomp ];then
      echo "running benchmark under ${1}-omp_aomp"
      cd ${1}-omp_aomp
      make -f Makefile.AMD run_time > log 2>&1
      cd ..
    fi
  else
    echo "the 2nd argument must be either NVIDIA or AMD"
    exit
  fi
fi
