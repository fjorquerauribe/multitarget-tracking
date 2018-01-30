#!/bin/bash
features=$4
pruning=$5

if [ "$pruning" == "nms" ];
then
    if [ "$features" == "raw" ];
    then
        # NMS RAW
        ../build/gm_phd -img data/$1/$2/$3/img1/000001.jpg -gt data/$1/$2/$3/gt/gt.txt -det data/raw_detections/CNN/$1/$2/$3.txt -pruning $5 -threshold $6 -neighbors $7 -minscores $8 -verbose $9
    elif [ "$features" == "cnn" ];
    then
        # NMS CNN
        ../build/gm_phd -img data/$1/$2/$3/img1/000001.jpg -gt data/$1/$2/$3/gt/gt.txt -det data/detections/CNN/$1/$2/$3.txt -pruning $5 -threshold $6 -neighbors $7 -minscores $8 -verbose $9
    elif [ "$features" == "frcnn" ];
    then
        # NMS FRCNN
        ../build/gm_phd -img data/$1/$2/$3/img1/000001.jpg -gt data/$1/$2/$3/gt/gt.txt -det data/detections/FRCNN/$1/$2/$3.txt -pruning $5 -threshold $6 -neighbors $7 -minscores $8 -verbose $9
    elif [ "$features" == "public" ];
    then
        # NMS public
        ../build/gm_phd -img data/$1/$2/$3/img1/000001.jpg -gt data/$1/$2/$3/gt/gt.txt -det data/$1/$2/$3/det/det.txt -pruning $5 -threshold $6 -neighbors $7 -minscores $8 -verbose $9
    else
        echo "No setting available"
    fi
elif [ "$pruning" == "dpp" ];
then   
    if [ "$features" == "raw" ];
    then
        # DPP RAW
        ../build/gm_phd -img data/$1/$2/$3/img1/000001.jpg -gt data/$1/$2/$3/gt/gt.txt -det data/raw_detections/CNN/$1/$2/$3.txt -pruning $5 -epsilon $6 -verbose $7
    elif [ "$features" == "cnn" ];
    then
        # DPP CNN
        ../build/gm_phd -img data/$1/$2/$3/img1/000001.jpg -gt data/$1/$2/$3/gt/gt.txt -det data/detections/CNN/$1/$2/$3.txt -pruning $5 -epsilon $6 -verbose $7
    elif [ "$features" == "frcnn" ];
    then
        # DPP FRCNN
        ../build/gm_phd -img data/$1/$2/$3/img1/000001.jpg -gt data/$1/$2/$3/gt/gt.txt -det data/detections/FRCNN/$1/$2/$3.txt -pruning $5 -epsilon $6 -verbose $7
    else
        echo "No setting available"
    fi
else
    if [ "$features" == "raw" ];
    then
        # No pruning public RAW
        ../build/gm_phd -img data/$1/$2/$3/img1/000001.jpg -gt data/$1/$2/$3/gt/gt.txt -det data/raw_detections/CNN/$1/$2/$3.txt -verbose $5
    elif [ "$features" == "cnn" ];
    then
        # No pruning public CNN
        ../build/gm_phd -img data/$1/$2/$3/img1/000001.jpg -gt data/$1/$2/$3/gt/gt.txt -det data/detections/CNN/$1/$2/$3.txt -verbose $5
    elif [ "$features" == "frcnn" ];
    then
        # No pruning public FRCNN
        ../build/gm_phd -img data/$1/$2/$3/img1/000001.jpg -gt data/$1/$2/$3/gt/gt.txt -det data/detections/FRCNN/$1/$2/$3.txt -verbose $5
    elif [ "$features" == "public" ];
    then
        # No pruning public public
        ../build/gm_phd -img data/$1/$2/$3/img1/000001.jpg -gt data/$1/$2/$3/gt/gt.txt -det data/$1/$2/$3/det/det.txt -verbose $5
    else
        echo "No setting available"
    fi
fi