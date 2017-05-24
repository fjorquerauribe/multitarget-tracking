#ifndef MTT_DPP_PETS_H
#define MTT_DPP_PETS_H

#include <time.h>
#include <iostream>
#include <cstdlib>

#include "utils/image_generator.hpp"
#include "utils/utils.hpp"
#include "dpp.hpp"
#include "models/phd_particle_filter.hpp"

#ifdef WITH_CUDA
#include "detectors/cuda_hog_detector.hpp"
#else
#include "detectors/hog_detector.hpp"
#endif

using namespace std;
using namespace cv;

class MultiTargetTrackingDPPPets
{
public:
	MultiTargetTrackingDPPPets();
	MultiTargetTrackingDPPPets(string _firstFrameFileName, string _groundTruthFileName,
	double _epsilon, double _mu, double _lambda, int _npart);
	void run();
private:
	string firstFrameFileName, groundTruthFileName;
	int npart;
	double epsilon, mu, lambda;
	ImageGenerator generator;
};

#endif