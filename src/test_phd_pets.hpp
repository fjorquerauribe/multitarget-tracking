#ifndef MTT_PHD_PETS_H
#define MTT_PHD_PETS_H

#include <time.h>
#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utils/image_generator.hpp"
#include "utils/utils.hpp"
#include "models/phd_particle_filter.hpp"

#ifdef WITH_CUDA
#include "detectors/cuda_hog_detector.hpp"
#else
#include "detectors/hog_detector.hpp"
#endif

using namespace std;
using namespace cv;

class MultiTargetTrackingPHDFilterPets
{
public:
	MultiTargetTrackingPHDFilterPets();
	MultiTargetTrackingPHDFilterPets(string _firstFrameFileName, string _groundTruthFileName, 
		int _group_threshold, double _hit_threshold, int _npart);
	void run();
private:
	//int numFrames;
	string firstFrameFileName, groundTruthFileName;
	int npart, group_threshold;
	double hit_threshold;
	ImageGenerator generator;
};

#endif