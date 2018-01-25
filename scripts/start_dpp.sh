#!/bin/bash
dataset=$1
type=$2
sequence=$3
epsilon=$4
mu=$5
lambda=$6
features=$7
verbose=$8

if [ "$features" == "raw" ];
then
    # raw features
    ../build/dpp -img data/$dataset/$type/$sequence/img1/000001.jpg -gt data/$dataset/$type/$sequence/gt/gt.txt \
    -det data/raw_detections/CNN/$dataset/$type/$sequence.txt -epsilon $epsilon -mu $mu -lambda $lambda -verbose $verbose
else
    # CNN features
    ../build/dpp -img data/$dataset/$type/$sequence/img1/000001.jpg -gt data/$dataset/$type/$sequence/gt/gt.txt \
    -det data/detections/CNN/$dataset/$type/$sequence.txt -epsilon $epsilon -mu $mu -lambda $lambda -verbose $verbose
fi
