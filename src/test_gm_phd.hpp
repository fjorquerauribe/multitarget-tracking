#ifndef GM_PHD_H
#define GM_PHD_H

#include <time.h>
#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>

#ifdef WITH_SEQUENTIAL_IMAGE_GENERATOR
	#include "utils/sequential_image_generator.hpp"
#else
	#include "utils/image_generator.hpp"
#endif

#include "utils/utils.hpp"
#include "models/phd_gaussian_mixture.hpp"

using namespace std;
using namespace cv;

class TestGMPHDFilter
{
public:
	TestGMPHDFilter();
	TestGMPHDFilter(string _firstFrameFileName, string _groundTruthFileName,
	 string _preDetectionFile);
	void run(bool verbose);
private:
	string firstFrameFileName, groundTruthFileName, preDetectionFile;
	
#ifdef WITH_SEQUENTIAL_IMAGE_GENERATOR
	SequentialImageGenerator generator;
#else
	ImageGenerator generator;
#endif
};

#endif