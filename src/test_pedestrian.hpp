#ifndef MTT_PEDESTRIAN_H
#define MTT_PEDESTRIAN_H

#include <time.h>
#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "utils/utils.hpp"
#include "models/phd_particle_filter.hpp"
#include "detectors/hog_detector.hpp"

using namespace std;
using namespace cv;

class TestPedestrian
{
public:
	TestPedestrian();
	TestPedestrian(int _npart);
	void run();
private:
	int npart;
};

#endif