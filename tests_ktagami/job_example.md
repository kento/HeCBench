# 使用メモリ・計算時間計測に使用したスクリプト例

## nvidia

### check.sh
```
#!/bin/sh
params1=" 100" # 3.5 GB
params2=" 100"
#
EXE=./main
#################
nvidia-smi --query-gpu=memory.used,memory.total --format=csv --loop-ms=1 > log.mem &
( time $EXE $params1  ) > Log 2>&1
pkill nvidia-smi
#################
./eval_mem_nv.sh >> Log

( time $EXE $params2  ) > Log2 2>&1
```

Log の最後の行に使用メモリが表示される
```
min max diff memory in [MB] : 2 3467 3465
```

## amd

### check.sh

```
#!/bin/sh
params1="  100"
params2="  100"
#
EXE=./main
#################
./rock-smi.sh > log.mem &#
( time $EXE $params1  ) > Log 2>&1
pkill ./rock-smi.sh
#################
./eval_mem_amd.sh >> Log

( time $EXE $params2  ) > Log2 2>&1
```

### eval_mem_amd.sh
```
#!/bin/sh
#maxm=$( cat log.mem | awk '{if(m<$8) m=$8} END{print m}' )
#maxm=$( printf %.2f $( echo ${maxm}/1000000 | bc -l ) )
maxm=$( cat log.mem | awk '{if(m<$8) m=$8} END{printf("%.2f\n", m/1000000)}' )

#minm=$( cat log.mem | awk 'BEGIN{m="inf"}{if(m>$8) m=$8} END{print m}' )
#minm=$( printf %.2f $( echo ${minm}/1000000 | bc -l ) )
minm=$( cat log.mem | awk 'BEGIN{m="inf"}{if(m>$8) m=$8} END{printf("%.2f\n", m/1000000)}' )

diffm=$( printf %.2f $( echo ${maxm}-${minm} | bc -l ) )
echo "min max diff memory in [MB] : ${minm} ${maxm} ${diffm}"
```
(甲賀さん版を少し変更してます)
