# log_run_bench.mem (仮)

run_mem モードで出力させた log_run_bench.std から log_run_bench.mem を作成する方法
log_run_bench.mem を保存しておけば、run モードで log_run_bench.std を上書きしても問題ない。

```
#!/bin/sh
#for n in `cat List_full`
for n in `cat List_part`
do
    num=0
    while [ $num -lt 5 ];
    do
        if [ $num = 0 ]; then dir=$n-cuda; fi
        if [ $num = 1 ]; then dir=$n-hip; fi
        if [ $num = 2 ]; then dir=$n-hipified; fi
        if [ $num = 3 ]; then dir=$n-omp_nvc; fi
        if [ $num = 4 ]; then dir=$n-omp_aomp; fi

        if [ -d $dir ]; then
            c1=`grep "min max diff" $dir/log_run_bench.std`
            if [ "c1" != "" ]; then
                echo $c1 > $dir/log_run_bench.mem
            fi
        fi
        let num="$num +1"
    done
done
```
