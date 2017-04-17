#ifndef MTT_PHD_PETS
#define MTT_PHD_PETS

#include <time.h>
#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "detectors/hog_detector.hpp"
#include "utils/image_generator.hpp"
#include "utils/utils.hpp"
#include "models/phd_particle_filter.hpp"

using namespace std;
using namespace cv;

class MultiTargetTrackingPHDFilterPets
{
public:
	MultiTargetTrackingPHDFilterPets();
	MultiTargetTrackingPHDFilterPets(string _firstFrameFileName, string _groundTruthFileName, 
		double group_threshold, double hit_threshold, int _npart);
	void run();
private:
	//int numFrames;
	string firstFrameFileName, groundTruthFileName;
	HOGDetector hogDetector;
	int npart;
	ImageGenerator generator;
};

#endif