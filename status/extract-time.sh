#!/bin/sh
echo "|" name "|" cuda "|" hip  "|" hipified "|" omp_nvc "|" omp_aomp "|"
echo "|" "--" "|" "--" "|" "--" "|" "--"     "|" "--"   "|" -- "|"
#
for n in `cat List_full`
do
    dir1=$n-cuda
    dir2=$n-hip
    dir3=$n-hipified
    dir4=$n-omp_nvc
    dir5=$n-omp_aomp
    
    if [ -d $dir1 ]; then
	c1=`grep Error $dir1/log.build | grep make`
	if [ "$c1"  = "" ]; then
	    c2=`grep -i error $dir1/log.err`
	    if [ "$c2"  = "" ]; then
		time1=`grep real $dir1/log.time | awk '{print $NF}'`
	    else
		time1="exe err"
	    fi
	else
	    time1="build err"
	fi
    else
	time1="--"
    fi

    if [ -d $dir2 ]; then
	c1=`grep Error $dir2/log.build | grep make`
	if [ "$c1" = "" ]; then
	    c2=`grep -i error $dir2/log.err`
	    if [ "$c2"  = "" ]; then
		time2=`grep real $dir2/log.time | awk '{print $NF}'`
	    else
		time2="exe err"
	    fi
	else
	    time2="build err"
	fi
    else
	time2="--"
    fi

    if [ -d $dir3 ]; then
	c1=`grep Error $dir3/log.build | grep make`
	if [ "$c1" = "" ]; then
	    c2=`grep -i error $dir3/log.err`
	    if [ "$c2"  = "" ]; then
		time3=`grep real $dir3/log.time | awk '{print $NF}'`
	    else
		time3="exe err"
	    fi
	else
	    time3="build err"
	fi
    else
	time3="--"
    fi

    if [ -d $dir4 ]; then
	c1=`grep Error $dir4/log.build | grep make`
	if [ "$c1" = "" ]; then
	    c2=`grep -i error $dir4/log.err`
	    if [ "$c2"  = "" ]; then
		time4=`grep real $dir4/log.time | awk '{print $NF}'`
	    else
		time4="exe err"
	    fi
	else
	    time4="build err"
	fi
	else
	time4="--"
    fi

    if [ -d $dir5 ]; then
	c1=`grep Error $dir5/log.build | grep make`
	if [ "$c1" = "" ]; then
	    c2=`grep -i error $dir5/log.err`
	    if [ "$c2"  = "" ]; then
		time5=`grep real $dir5/log.time | awk '{print $NF}'`
	    else
		time5="exe err"
	    fi
	else
	    time5="build err"
	fi
    else
	time5="--"
    fi
    #
    echo "|" $n "|" $time1 "|" $time2 "|" $time3 "|" $time4 "|" $time5 "|"
done
