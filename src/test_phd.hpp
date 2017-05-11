#ifndef MTT_PHD_PF_H
#define MTT_PHD_PF_H

#include <time.h>
#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "utils/image_generator.hpp"
#include "utils/utils.hpp"
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