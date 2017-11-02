declare -a datasets=(2DMOT2015 MOT16)

for dataset in "${datasets[@]}"
do
    while read sequence;
    do
        echo $dataset,$sequence
        /bin/bash $PWD/../build/start_phd.sh $dataset train $sequence 10 > ../build/phd_results/$dataset/train/$sequence.txt
    done <./data/$dataset/train/sequences.lst
done
