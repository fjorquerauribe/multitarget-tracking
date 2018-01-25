#!/bin/bash

declare -a datasets=(MOT16)
#declare -a datasets=(2DMOT2015)
#declare -a datasets=(MOT17)
OUTPUT_DIRECTORY=$1

for dataset in "${datasets[@]}"
do
    while read sequence;
    do
        mkdir -p ../build/results/$OUTPUT_DIRECTORY/$dataset/train/
        echo $dataset,$sequence
        /bin/bash $PWD/start_gm.sh $dataset train $sequence $OUTPUT_DIRECTORY  0 > ../build/results/$OUTPUT_DIRECTORY/$dataset/train/$sequence.txt
    done <./data/$dataset/sequences.lst
done
