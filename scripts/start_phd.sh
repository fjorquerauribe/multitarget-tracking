#!/bin/bash
detector=$5

if [ "$detector" == "yolo" ];
then
    ../build/phd -img data/$1/$2/$3/img1/000001.jpg -gt data/$1/$2/$3/gt/gt.txt -config data/yolo/yolov3.cfg -model data/yolo/yolov3.weights -classes data/yolo/coco.names -min_confidence $6 -npart $4
else
    ../build/phd -img data/$1/$2/$3/img1/000001.jpg -gt data/$1/$2/$3/gt/gt.txt -det data/$1/$2/$3/det/det.txt -npart $4
fi
