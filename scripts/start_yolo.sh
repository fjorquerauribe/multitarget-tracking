../build/yolo -img data/$1/$2/$3/img1/000001.jpg -gt data/$1/$2/$3/gt/gt.txt -config $PWD/yolov3.cfg -model $PWD/yolov3.weights  -classes $PWD/coco.names -min_confidence 0.5 -epsilon $4 -verbose $5