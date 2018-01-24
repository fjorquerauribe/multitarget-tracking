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

typedef struct{
	int label;
  	Scalar color;
	Rect bbox;
	double survival_rate;
	VectorXd feature;
	//int conf;
} Target;

class Utils{
public:
	static int** compute_cost_matrix(vector<Target> tracks, vector<Target> new_tracks);
	static int** compute_overlap_matrix(vector<Target> tracks, vector<Target> new_tracks);
	static void detections_quality(VectorXd &detections_weights, vector<Rect> detections, 
		vector<Target> tracks, VectorXd &contain, double overlap_threshold, double lambda);
};

#endif