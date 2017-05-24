#!/bin/bash
declare -a epsilon="0.1 0.3 0.5"
declare -a mu="0.3 0.5 0.7 0.8"
declare -a lambda="-0.1 0.1 0.01"

for e in $epsilon
do
	for m in $mu
	do
		for l in $lambda
		do
			echo "frame,x,y,width,height" | tee output-dpp-s2l1-$e-$m-$l.txt && /bin/bash $PWD/../build/start_dpp_mot15.sh PETS09-S2L1 $e $m $l 100 | tee -a output-dpp-s2l1-$e-$m-$l.txt
		done
	done
done
