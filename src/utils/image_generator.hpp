#ifndef IMAGE_GENERATOR_H
#define IMAGE_GENERATOR_H

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <utils.hpp>

using namespace std;
using namespace cv;

class ImageGenerator{
public:
  ImageGenerator();
  ImageGenerator(string _firstFrameFilename, string _groundTruthFile);
  ImageGenerator(string _firstFrameFilename, string _groundTruthFile, string _detectionsFile);
  bool hasEnded();
  void moveNext();
  Mat getFrame(int frame_num);
  vector<Rect> getDetections(int frame_num);
  vector<Target> getGroundTruth(int frame_num);
  size_t getDatasetSize();
private:
  void readDetections(string str);
  void readGroundTruth(string str, string dataset = "mot");
  int frame_id;
  void getNextFilename(string& filename, string dataset = "mot");
  vector<Mat> images;
  vector< vector<Target> > ground_truth;
  vector< vector<Rect> > detections;
};

#endif // IMAGE_GENERATOR_H
