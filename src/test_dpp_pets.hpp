#ifndef MTT_DPP_PETS_H
#define MTT_DPP_PETS_H

#include <time.h>
#include <iostream>
#include <cstdlib>

#include "utils/image_generator.hpp"
#include "detectors/hog_detector.hpp"
#include "dpp.hpp"

#include "models/phd_particle_filter.hpp"

using namespace std;
using namespace cv;

class MultiTargetTrackingDPPPets
{
public:
	MultiTargetTrackingDPPPets();
	MultiTargetTrackingDPPPets(string _firstFrameFileName, string _groundTruthFileName, int _npart);
	void run();
private:
	string firstFrameFileName, groundTruthFileName;
	int npart;
	HOGDetector hogDetector;
	DPP dpp;
	ImageGenerator generator;
};

#endif