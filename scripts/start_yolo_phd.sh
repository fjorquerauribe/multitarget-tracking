../build/yolo -img data/$1/$2/$3/img1/000001.jpg -gt data/$1/$2/$3/gt/gt.txt -config ./yolov3.cfg -model ./yolov3.weights  -classes ./coco.names -min_confidence 0.95 -epsilon $4 -verbose $5
