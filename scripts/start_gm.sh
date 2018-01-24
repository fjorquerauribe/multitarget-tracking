#!/bin/bash
features=$4

if [ "$features" == "raw" ];
then
    # raw features
    ../build/gm_phd -img $1/$2/$3/img1/000001.jpg -gt $1/$2/$3/gt/gt.txt -det data/RAW/$1/$2/$3.txt  -verbose $5
elif [ "$features" == "cnn" ];
then
    # CNN features
    ../build/gm_phd -img $1/$2/$3/img1/000001.jpg -gt $1/$2/$3/gt/gt.txt -det data/CNN/$1/$2/$3.txt  -verbose $5
else
    # public features
    ../build/gm_phd -img $1/$2/$3/img1/000001.jpg -gt $1/$2/$3/gt/gt.txt -det $1/$2/$3/det/det.txt  -verbose $5

fi