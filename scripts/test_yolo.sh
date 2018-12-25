#declare -a datasets=(2DMOT2015 MOT16)
declare -a datasets=(2DMOT2015)
declare -a types=(train)
#declare -a types=(train test)
declare -a epsilon=(0.9)
declare -a method=(nms)
OUTPUT_DIRECTORY=$1

for dataset in "${datasets[@]}"
do
    for type in "${types[@]}"
    do
        for e in "${epsilon[@]}"
        do
	    for m in "${method[@]}"
		do
	            mkdir -p ~/code/python/py-motmetrics/motmetrics/data/$dataset/$method/$type/$e/
        	    ls ./data/$dataset/$type/ > ./data/$dataset/$type/sequences.lst
            	    sed -i '/sequences.lst/d' ./data/$dataset/$type/sequences.lst
            	    while read sequence;
            	    do
                	echo $dataset,$sequence,$e
                	/bin/bash $PWD/start_yolo_$method.sh $dataset $type $sequence $e 0 > ~/code/python/py-motmetrics/motmetrics/data/$dataset/$method/$type/$e/$sequence.txt
            	    done <./data/$dataset/$type/sequences.lst
        	done
	done
    done
done
