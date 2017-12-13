#declare -a datasets=(2DMOT2015 MOT16)
declare -a datasets=(2DMOT2015)
declare -a hit_threshold=(0.4)
declare -a group_threshold=(0.0)

for dataset in "${datasets[@]}"
do
    for hit in "${hit_threshold[@]}"
    do
        for gt in "${group_threshold[@]}"
        do
            mkdir -p results/$dataset/lr_detector/LR_Mars_Model
            while read sequence;
            do
                echo $dataset,$sequence
                /bin/bash $PWD/../build/start_lr_detector.sh $dataset train $sequence LR_Mars_Model $gt $hit > ../build/results/$dataset/lr_detector/LR_Mars_Model/$sequence.txt
            done <./data/$dataset/train/sequences.lst
        done
    done
done