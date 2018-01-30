#!/bin/bash
features=$4

if [ "$features" == "raw" ];
then
    # raw features
    ../build/gm_phd -img data/$1/$2/$3/img1/000001.jpg -gt data/$1/$2/$3/gt/gt.txt -det data/raw_detections/CNN/$1/$2/$3.txt  -verbose $5
elif [ "$features" == "cnn" ];
then
    # CNN features
    ../build/gm_phd -img data/$1/$2/$3/img1/000001.jpg -gt data/$1/$2/$3/gt/gt.txt -det data/detections/CNN/$1/$2/$3.txt  -verbose $5
elif [ "$features" == "frcnn" ];
then
    # FRCNN features
    ../build/gm_phd -img data/$1/$2/$3/img1/000001.jpg -gt data/$1/$2/$3/gt/gt.txt -det data/detections/FRCNN/$1/$2/$3.txt  -verbose $5
elif [ "$features" == "frcnn-deepsort" ];
then
    # FRCNN DEEP SORT features
    ../build/gm_phd -img data/$1/$2/$3/img1/000001.jpg -gt data/$1/$2/$3/gt/gt.txt -det data/detections/FRCNN-DEEPSORT/$1/$2/$3.txt  -verbose $5
elif [ "$features" == "hog" ];
then
    # HOG features
    ../build/gm_phd -img data/$1/$2/$3/img1/000001.jpg -gt data/$1/$2/$3/gt/gt.txt -det data/detections/HOG/$1/$2/$3.txt  -verbose $5
else
    # public features
    ../build/gm_phd -img data/$1/$2/$3/img1/000001.jpg -gt data/$1/$2/$3/gt/gt.txt -det data/$1/$2/$3/det/det.txt  -verbose $5
fi