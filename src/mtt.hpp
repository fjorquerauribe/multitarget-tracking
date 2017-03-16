#ifndef MULTITARGET_TRACKING_H
#define MULTITARGET_TRACKING_H

#include <time.h>
#include <iostream>
#include <cstdlib>

#include "utils/image_generator.hpp"
#include "detectors/hog_detector.hpp"
#include "dpp.hpp"

#include "models/phd_particle_filter.hpp"

using namespace std;
using namespace cv;

class MultiTargetTracking
{
public:
	MultiTargetTracking();
	MultiTargetTracking(string _firstFrameFileName, string _groundTruthFileName, string _preDetectionFile, int _npart);
	void run();
private:
	//int numFrames;
	string firstFrameFileName, groundTruthFileName, preDetectionFile;
	int npart;
	bool initialized;
	HOGDetector hogDetector;
	DPP dpp;
	ImageGenerator generator;
};

#endif