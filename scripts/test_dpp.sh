declare -a datasets=(2DMOT2015 MOT16)
declare -a epsilon=(0.1)
declare -a mu=(-0.1)
declare -a lambda=(0.2 0.4 0.6 0.8 1.0)

for dataset in "${datasets[@]}"
do
    for e in "${epsilon[@]}"
    do
        for m in "${mu[@]}"
        do
            for l in "${lambda[@]}"
            do
                mkdir -p ../build/dpp_results/$e-$m-$l/$dataset/train/
                while read sequence;
                do
                    echo $dataset,$sequence,$e,$m,$l
                    /bin/bash $PWD/../build/start_dpp.sh $dataset train $sequence $e $m $l 10 > ../build/dpp_results/$e-$m-$l/$dataset/train/$sequence.txt
                done <./data/$dataset/train/sequences.lst
            done
        done
    done
done
