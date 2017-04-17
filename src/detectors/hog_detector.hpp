#ifndef HOG_DETECTOR_H
#define HOG_DETECTOR_H

#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Core>
#include <iostream>

using namespace cv;
using namespace std;
using namespace Eigen;

class HOGDetector
{
public:
	HOGDetector();
	HOGDetector(double group_threshold, double hit_threshold);
	vector<Rect> detect(Mat &frame);
	void draw();
	//vector<Rect> getDetections();
	MatrixXd getFeatureValues();
	VectorXd getDetectionWeights();

private:
	HOGDescriptor hog;
	vector<Rect> detections;
	VectorXd weights;
	Mat frame;
	double group_threshold, hit_threshold;
};

#endif