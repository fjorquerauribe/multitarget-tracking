declare -a datasets=(2DMOT2015)
declare -a epsilon=(0.1 0.3 0.5 0.7)
declare -a mu=(0.1 0.3 0.5 0.7)
declare -a lambda=(0.1 0.3 0.5 0.7 0.9)
OUTPUT_DIRECTORY=$1

for dataset in "${datasets[@]}"
do
    for e in "${epsilon[@]}"
    do
        for m in "${mu[@]}"
        do
            for l in "${lambda[@]}"
            do
                mkdir -p ../build/$OUTPUT_DIRECTORY/$e-$m-$l/$dataset/train/
                while read sequence;
                do
                    echo $dataset,$sequence,$e,$m,$l
                    /bin/bash $PWD/../build/start_dpp.sh $dataset train $sequence $e $m $l 10 > ../build/$OUTPUT_DIRECTORY/$e-$m-$l/$dataset/train/$sequence.txt
                done <./data/$dataset/train/sequences.lst
            done
        done
    done
done
