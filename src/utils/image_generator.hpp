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

#include <opencv2/core.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace Eigen;

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
  MatrixXd getFeatures(int frame_num);
  size_t getDatasetSize();
private:
  void readDetections(string str);
  void readGroundTruth(string str);
  int frame_id;
  void getNextFilename(string& filename);
  vector<Mat> images;
  vector< vector<Target> > ground_truth;
  vector< vector<Rect> > detections;
  vector< MatrixXd > features;
};

#endif // IMAGE_GENERATOR_H
