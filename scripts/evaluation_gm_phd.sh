declare -a datasets=(MOT16)
declare -a types=(train)
OUTPUT_DIRECTORY=$2

# DPP
declare -a dpp_features=(frcnn cnn raw)
declare -a epsilon=(0.1 0.3 0.5 0.7 0.9 0.95)

for dataset in "${datasets[@]}"
do
    for type in "${types[@]}"
    do
        ls ./data/$dataset/$type/ > ./data/$dataset/$type/sequences.lst
        sed -i '/sequences.lst/d' ./data/$dataset/$type/sequences.lst
        for feat in "${dpp_features[@]}"
        do
            for eps in "${epsilon[@]}"
            do
                mkdir -p ../build/results/$OUTPUT_DIRECTORY/dpp/$feat/$eps/$dataset/$type/
                while read sequence;
                do
                    echo $dataset,$type,$feat,$eps,$sequence
                    /bin/bash $PWD/start_gm_phd.sh $dataset $type $sequence $feat dpp $eps 0 > ../build/results/$OUTPUT_DIRECTORY/dpp/$feat/$eps/$dataset/$type/$sequence.txt
                done <./data/$dataset/$type/sequences.lst
            done
            
        done
        rm ./data/$dataset/$type/sequences.lst
    done
done

# NMS
declare -a nms_features=(frcnn cnn raw public)
declare -a threshold=(0.1 0.3 0.5 0.7 0.9)
declare -a neighbors=(0 1 2 3 4)
declare -a min_score_sum=(0.0 0.1 0.3 0.5 0.7 0.9)

for dataset in "${datasets[@]}"
do
    for type in "${types[@]}"
    do
        ls ./data/$dataset/$type/ > ./data/$dataset/$type/sequences.lst
        sed -i '/sequences.lst/d' ./data/$dataset/$type/sequences.lst
        for feat in "${nms_features[@]}"
        do
            for thold in "${threshold[@]}"
            do
                for nbr in "${neighbors}"
                do
                    for score in "${min_score_sum[@]}"
                    do
                        mkdir -p ../build/results/$OUTPUT_DIRECTORY/nms/$feat/$thold/$nbr/$score/$dataset/$type/
                        while read sequence;
                        do
                            echo $dataset,$type,$feat,$thold,$nbr,$score,$sequence
                            /bin/bash $PWD/start_gm_phd.sh $dataset $type $sequence $feat nms $thold $nbr $score 0 > ../build/results/$OUTPUT_DIRECTORY/nms/$feat/$thold/$nbr/$score/$dataset/$type/$sequence.txt
                        done <./data/$dataset/$type/sequences.lst
                    done
                done
            done            
        done
        rm ./data/$dataset/$type/sequences.lst
    done
done

# No pruning
declare -a no_pruning_features=(frcnn cnn raw public)

for dataset in "${datasets[@]}"
do
    for type in "${types[@]}"
    do
        ls ./data/$dataset/$type/ > ./data/$dataset/$type/sequences.lst
        sed -i '/sequences.lst/d' ./data/$dataset/$type/sequences.lst
        for feat in "${no_pruning_features[@]}"
        do
            mkdir -p ../build/results/$OUTPUT_DIRECTORY/no_pruning/$feat/$eps/$dataset/$type/
            while read sequence;
            do
                echo $dataset,$type,$feat,$sequence
                /bin/bash $PWD/start_gm_phd.sh $dataset $type $sequence $feat no_pruning 0 > ../build/results/$OUTPUT_DIRECTORY/no_pruning/$feat/$dataset/$type/$sequence.txt
            done <./data/$dataset/$type/sequences.lst
        done
        rm ./data/$dataset/$type/sequences.lst
    done
done