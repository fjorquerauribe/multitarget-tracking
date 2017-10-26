declare -a datasets=(2DMOT2015 MOT16)
declare -a types=(test train)
declare -a scripts=(start_phd_mot15.sh start_phd_mot16.sh)

for script in "${scripts[@]}"
do
    for dataset in "${datasets[@]}"
    do
        for type in "${types[@]}"
        do
            while read sequence;
            do
            /bin/bash $PWD/../build/$script $type $sequence 50 > ../build/phd_results/$dataset/$type/$sequence.txt
            done <./data/$dataset/$type/sequences.lst
        done
    done
done