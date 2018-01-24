#ifndef SEQUENTIAL_IMAGE_GENERATOR_H
#define SEQUENTIAL_IMAGE_GENERATOR_H

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include "utils.hpp"

#include <opencv2/core.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace Eigen;

class SequentialImageGenerator{
public:
  SequentialImageGenerator();
  SequentialImageGenerator(string _firstFrameFilename, string _groundTruthFile, string _detectionsFile);
  bool hasEnded();
  void moveNext();
  Mat getFrame(int frame_num);
  vector<Target> getGroundTruth(int frame_num);
  vector<Rect> getDetections();
  VectorXd getDetectionWeights();
  MatrixXd getDetectionFeatures();
  size_t getDatasetSize();
  void readDetections(int frame);

private:
  void readGroundTruth(string str);
  int frame_id;
  void getNextFilename(string& filename);
  vector<Mat> images;
  
  string detectionsFileName;
  ifstream dt_file;
  int pointerPos;

  vector< vector<Target> > ground_truth;
  vector<Rect> detections;
  VectorXd detection_weights;
  MatrixXd features;
};

#endif // IMAGE_GENERATOR_H
