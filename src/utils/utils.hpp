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

using namespace std;
using namespace cv;

typedef struct{
	int label;
  	Scalar color;
	Rect bbox;
	//int conf;
} Target;

class Utils{
public:
	static int** compute_cost_matrix(vector<Target> tracks, vector<Target> new_tracks);
};

#endif