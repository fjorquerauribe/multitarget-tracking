#ifndef IMAGE_GENERATOR_H
#define IMAGE_GENERATOR_H

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

using namespace std;
using namespace cv;

typedef struct{
	int label;
	Rect bbox;
} Target;

class ImageGenerator{
public:
  ImageGenerator();
  ImageGenerator(string _firstFrameFilename, string _groundTruthFile, string _detectionsFile);
  bool hasEnded();
  void moveNext();
  Mat getFrame(int frame_num);
  vector<Rect> getDetections(int frame_num);
  vector<Target> getGroundTruth(int frame_num);
  int getDatasetSize();
private:
  void readDetections(string str);
  void readGroundTruth(string str);
  int frame_id;
  void getNextFilename(string& filename);
  vector<Mat> images;
  vector< vector<Target> > ground_truth;
  vector< vector<Rect> > detections;
};

#endif // IMAGE_GENERATOR_H
