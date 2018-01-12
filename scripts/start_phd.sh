#../build/phd -img data/$1/$2/$3/img1/000001.jpg -gt data/$1/$2/$3/gt/gt.txt -det ../data/det/$1/$2/$3.txt -npart $4
#../build/phd -img data/$1/$2/$3/img1/000001.jpg -gt data/$1/$2/$3/gt/gt.txt -det data/$1/$2/$3/det/det.txt -npart $4
../build/phd -img data/$1/$2/$3/img1/000001.jpg -gt data/$1/$2/$3/gt/gt.txt -det data/DETS_HOG/$1/$2/$3/det/det.txt -npart $4