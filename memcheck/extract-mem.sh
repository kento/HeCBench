#!/bin/sh
echo "# used memory [GB]"
echo ""
echo "|" name "|" cuda "|" hip  "|" hipified "|" omp_nvc "|" omp_aomp "|"
echo "|" "--" "|" "--" "|" "--" "|" "--"     "|" "--"   "|" -- "|"
#
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
	
	if [ $num = 0 ]; then comment="| "$n" |"; fi

	if [ -d $dir ]; then
	    c1=`grep Error $dir/log_run_bench.err | grep make`
	    if [ -e $dir/log.build ]; then
		c1=`grep Error $dir/log.build | grep make`
	    fi
	    if [ "$c1"  = "" ]; then
		c2=`grep -i error $dir/log.err`
		if [ "$c2"  = "" ]; then
		    if [ -e $dir/log_run_bench.mem ]; then
			mem=`cat $dir/log_run_bench.mem | awk '{printf("%5.1f\n",$(NF-1)/1E3)}'`
		    else
			mem="unknown"
		    fi
		else
		    mem="exe err"
		fi
	    else
		mem="build err"
	    fi
	else
	    mem="--"
	fi
	#
	comment=$comment" "$mem" |"
	let num="$num +1"
    done
    echo $comment
done
