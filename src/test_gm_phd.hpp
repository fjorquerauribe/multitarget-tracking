#ifndef GM_PHD_H
#define GM_PHD_H

#include <time.h>
#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>

#include "utils/image_generator.hpp"
#include "utils/utils.hpp"
#include "models/phd_gaussian_mixture.hpp"

using namespace std;
using namespace cv;

class TestGMPHDFilter
{
public:
	TestGMPHDFilter();
	TestGMPHDFilter(string _firstFrameFileName, string _groundTruthFileName,
	 string _detectionFile);
	void run(bool verbose);
	void run(bool verbose, PHDGaussianMixture filter);
	void run(bool verbose, double epsilon);
	void run(bool verbose, double threshold, int neighbors, double min_scores_num);
private:
	string firstFrameFileName, groundTruthFileName, detectionFile, pruningMethod;	
	ImageGenerator generator;
};

#endif