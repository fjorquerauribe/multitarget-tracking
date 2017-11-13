#ifndef MTT_LR_DETECTOR_H
#define MTT_LR_DETECTOR_H

#include <time.h>
#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "utils/image_generator.hpp"
#include "utils/c_utils.hpp"
#include "utils/utils.hpp"
#include "detectors/CPU_LR_hog_detector.hpp"

using namespace std;
using namespace cv;

class TestLRDetector
{
public:
	TestLRDetector();
	TestLRDetector(string _firstFrameFileName, string _groundTruthFileName,
	 string _modelFilesPath, double _group_threshold, double _hit_threshold);
	void run();
private:
	string firstFrameFileName, groundTruthFileName, modelFilesPath;
	ImageGenerator generator;
	double group_threshold, hit_threshold;
};

#endif