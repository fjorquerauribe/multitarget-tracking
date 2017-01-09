#ifndef MULTITARGET_TRACKING_H
#define MULTITARGET_TRACKING_H

#include <time.h>
#include <iostream>
#include <cstdlib>

#include "utils/image_generator.hpp"
#include "utils/cnn_reader.hpp"
#include "detectors/hog_detector.hpp"
#include "dpp.hpp"

using namespace std;
using namespace cv;

class MultiTargetTracking
{
public:
	MultiTargetTracking();
	MultiTargetTracking(string _firstFrameFileName, string _groundTruthFileName);
	MultiTargetTracking(string _firstFrameFileName, string _gtFileName, string _firstCNNFeaturesFile, string _firstPreDetectionFile);
	void run();
private:
	//int numFrames;
	string firstFrameFileName, groundTruthFileName, firstCNNFeaturesFile, firstPreDetectionFile;
	bool initialized;
	HOGDetector hogDetector;
	DPP dpp;
	ImageGenerator generator;
	CNNReader cnnReader;
};

#endif