declare -a datasets=(MOT16)
declare -a types=(train)
FEATURE=$1
OUTPUT_DIRECTORY=$2

for dataset in "${datasets[@]}"
do
    for type in "${types[@]}"
    do
        mkdir -p ../build/results/$OUTPUT_DIRECTORY/$dataset/$type/
        ls ./data/$dataset/$type/ > ./data/$dataset/$type/sequences.lst
        sed -i '/sequences.lst/d' ./data/$dataset/$type/sequences.lst
        while read sequence;
        do
            echo $dataset,$sequence
            /bin/bash $PWD/start_gm_phd.sh $dataset $type $sequence $FEATURE dpp 0.95 0 > ../build/results/$OUTPUT_DIRECTORY/$dataset/$type/$sequence.txt
        done <./data/$dataset/$type/sequences.lst
    done
done
