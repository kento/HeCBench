#!/bin/bash

if [ -z "${1}" ];then
  echo Usage : ./scripts/exec_run_bench.sh BENCHMARK_LIST_FILE GPU_TYPE
  exit
fi

GPU=${2}
if [ -z "${GPU}" ];then
  echo Usage : ./scripts/exec_run_bench.sh BENCHMARK_LIST_FILE GPU_TYPE
  exit
fi
run_mem=${3}
while read line
do echo ${line}
  ./scripts/run_bench.sh ${line} ${GPU} ${run_mem}
done < "${1}"

