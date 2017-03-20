#ifndef MTT_PHD_PF
#define MTT_PHD_PF

#include <time.h>
#include <iostream>
#include <cstdlib>

#include "utils/image_generator.hpp"
#include "models/phd_particle_filter.hpp"

using namespace std;
using namespace cv;

class MultiTargetTrackingPHDFilter
{
public:
	MultiTargetTrackingPHDFilter();
	MultiTargetTrackingPHDFilter(string _firstFrameFileName, string _groundTruthFileName, string _preDetectionFile, int _npart);
	void run();
private:
	//int numFrames;
	string firstFrameFileName, groundTruthFileName, preDetectionFile;
	int npart;
	ImageGenerator generator;
};

#endif