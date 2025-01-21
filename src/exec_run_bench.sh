#!/bin/bash

if [ -z "${1}" ];then
  echo Usage : ./exec_run_bench.sh BENCHMARK_LIST_FILE GPU_TYPE
  exit
fi

GPU=${2}
if [ -z "${GPU}" ];then
  echo Usage : ./exec_run_bench.sh BENCHMARK_LIST_FILE GPU_TYPE
  exit
fi
while read line
do echo ${line}
  ./run_bench.sh ${line} ${GPU}
done < "${1}"

