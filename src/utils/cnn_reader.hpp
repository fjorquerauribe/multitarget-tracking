#ifndef CNN_READER_H
#define CNN_READER_H

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace Eigen;

class CNNReader{

public:
	CNNReader();
	CNNReader(string _firstCNNFeaturesFile, string _firstPreDetectionFile);
	MatrixXd getFeatureValues();
	vector<Rect> getPreDetections();
	VectorXd getDetectionWeights();

private:
	void getNextCNNfeaturesFileName();
	void getNextPreDetectionsFileName();
	void readCNNfeatures();
	void readPreDetections();
	vector<Rect> preDetections;
	MatrixXd CNNfeatures;
	VectorXd detectionWeights;
	string preDetectionsFileName, CNNfeaturesFileName;

};

#endif