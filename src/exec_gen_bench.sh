#!/bin/bash

if [ -z "${1}" ];then
  echo Usage : ./exec_gen_bench.sh BENCHMARK_LIST_FILE
  exit
fi
while read line
do echo ${line}
  ./gen_bench.sh ${line}
done < "${1}"

