#!/bin/sh
echo "|" name "|" cuda "|" hip  "|" hipified "|" omp_nvc "|" omp_aomp "|"
echo "|" "--" "|" "--" "|" "--" "|" "--"     "|" "--"   "|" -- "|"
#
for n in `cat List_full`
do
    num=0
    while [ $num -lt 5 ];
    do
	if [ $num = 0 ]; then dir=$n-cuda; fi
	if [ $num = 1 ]; then dir=$n-hip; fi
	if [ $num = 2 ]; then dir=$n-hipified; fi
	if [ $num = 3 ]; then dir=$n-omp_nvc; fi
	if [ $num = 4 ]; then dir=$n-omp_aomp; fi
	
	if [ $num = 0 ]; then comment="| "$n" |"; fi

	if [ -d $dir ]; then
	    c1=`grep Error $dir/log_run_bench.err | grep make`
	    if [ -e $dir/log.build ]; then
		c1=`grep Error $dir/log.build | grep make`
	    fi
	    if [ "$c1"  = "" ]; then
		c2=`grep -i error $dir/log.err`
                c3=`grep "core dumped" $dir/log.time`
                if [ "$c2"  = "" -a "$c3" = "" ]; then
		    time=`grep real $dir/log.time | awk '{print $NF}'`
		else
		    time="exe err"
		fi
	    else
		time="build err"
	    fi
	else
	    time="--"
	fi
	#
	comment=$comment" "$time" |"
	let num="$num +1"
    done
    echo $comment
done
