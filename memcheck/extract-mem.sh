#!/bin/sh
echo "# used memory [GB]"
echo ""
echo "|" name "|" cuda "|" hip  "|" hipified "|" omp_nvc "|" omp_aomp "|"
echo "|" "--" "|" "--" "|" "--" "|" "--"     "|" "--"   "|" -- "|"
#
#SRCDIR=../src2
SRCDIR=./
#
for n in `cat List_full`
#for n in `cat List_part`
do
    num=0
    while [ $num -lt 5 ];
    do
        if [ $num = 0 ]; then dir=$SRCDIR/$n-cuda; fi
        if [ $num = 1 ]; then dir=$SRCDIR/$n-hip; fi
        if [ $num = 2 ]; then dir=$SRCDIR/$n-hipified; fi
        if [ $num = 3 ]; then dir=$SRCDIR/$n-omp_nvc; fi
        if [ $num = 4 ]; then dir=$SRCDIR/$n-omp_aomp; fi
	
	if [ $num = 0 ]; then comment="| "$n" |"; fi

	mem=""
	if [ -d $dir ]; then
	    c1=`grep Error $dir/log_run_bench.err | grep make`
            c1a=`echo $c1 | awk '{print $NF}'`
	    if [ -e $dir/log.build ]; then
		c1=`grep Error $dir/log.build | grep make`
	    fi
	    if [ "$c1"  = "" ]; then
		c2=`grep -i error $dir/log.err`
                c3=`grep "core dumped" $dir/log.time`

                c4=`grep Terminated $dir/log_run_bench.err`
		c5=`grep "timeout was set" $dir/log_run_bench.err`

		if [ "$c2"  = "" -a "$c3" = "" ]; then
		    if [ -e $dir/log_run_bench.mem ]; then
			mem=`cat $dir/log_run_bench.mem | awk '{printf("%5.1f\n",$(NF-1)/1E3)}'`
		    else
			mem="unknown"
		    fi
#####################
		    pushd $dir >& /dev/null
		    if [ $num -eq 0 -o $num -eq 3 ]; then
			../scripts/m.sh > log_run_bench.mem
		    else
			../scripts/rocm-m.sh > log_run_bench.mem
		    fi
		    popd  >& /dev/null

		    mem=`cat $dir/log_run_bench.mem | awk '{printf("%5.1f\n",$(NF-1)/1E3)}'`
#####################
		else
		    mem="exe err"
		fi
	    else
                if [ $c1a -gt 100 ]; then
                    mem="exe err"
                else
                    mem="build err"
                fi
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
