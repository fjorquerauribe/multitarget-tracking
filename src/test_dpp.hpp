#ifndef MTT_DPP_H
#define MTT_DPP_H

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

class TestDPP
{
public:
	TestDPP();
	TestDPP(string _firstFrameFileName, string _groundTruthFileName, string _preDetectionFile,
		double _epsilon, double _mu, double _lambda, int _npart);
	void run();
private:
	//int numFrames;
	string firstFrameFileName, groundTruthFileName, preDetectionFile;
	int npart;
	double epsilon, mu, lambda;
	ImageGenerator generator;
};

#endif