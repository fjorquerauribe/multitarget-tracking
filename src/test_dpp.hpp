#ifndef MTT_DPP_H
#define MTT_DPP_H

#include <time.h>
#include <iostream>
#include <cstdlib>

#include "utils/image_generator.hpp"
#include "detectors/hog_detector.hpp"
#include "dpp.hpp"

#include "models/phd_particle_filter.hpp"

using namespace std;
using namespace cv;

class MultiTargetTrackingDPP
{
public:
	MultiTargetTrackingDPP();
	MultiTargetTrackingDPP(string _firstFrameFileName, string _groundTruthFileName, string _preDetectionFile, int _npart);
	void run();
private:
	//int numFrames;
	string firstFrameFileName, groundTruthFileName, preDetectionFile;
	int npart;
	HOGDetector hogDetector;
	DPP dpp;
	ImageGenerator generator;
};

#endif