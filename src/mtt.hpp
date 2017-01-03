#ifndef MULTITARGET_TRACKING_H
#define MULTITARGET_TRACKING_H

#include <time.h>
#include <iostream>
#include <cstdlib>

#include "utils/image_generator.hpp"
#include "detectors/hog_detector.hpp"
#include "dpp.hpp"

using namespace std;
using namespace cv;

class MultiTargetTracking
{
public:
	MultiTargetTracking();
	MultiTargetTracking(string _firstFrameFileName, string _groundTruthFileName);
	void run();
private:
	int numFrames;
	string firstFrameFileName, groundTruthFileName;
	bool initialized;
	HOGDetector hogDetector;
	DPP dpp;
};

#endif