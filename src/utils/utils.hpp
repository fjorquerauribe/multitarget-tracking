#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace Eigen;

struct MyTarget{
	int label;
  	Scalar color;
	Rect bbox;
	double survival_rate;
	VectorXd feature;
	double score;
};

class Utils{
public:
	static int** compute_cost_matrix(vector<MyTarget> tracks, vector<MyTarget> new_tracks, double Ql, double Qs);
	static int** compute_affinity_matrix(vector<MyTarget> tracks, vector<MyTarget> new_tracks);
	static int** compute_overlap_matrix(vector<MyTarget> tracks, vector<MyTarget> new_tracks);
	static void detections_quality(VectorXd &detections_weights, vector<Rect> detections, 
	vector<MyTarget> tracks, VectorXd &contain, double overlap_threshold, double lambda);
};

#endif