#ifndef MTT_PHD_PF_H
#define MTT_PHD_PF_H

#include <time.h>
#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "utils/image_generator.hpp"
#include "utils/utils.hpp"
#include "models/phd_particle_filter.hpp"
#include "detectors/yolo_detector.hpp"
using namespace std;
using namespace cv;

class TestPHDFilter
{
public:
	TestPHDFilter();
	TestPHDFilter(string _firstFrameFileName, string _groundTruthFileName,
	 	string _preDetectionFile, int _npart);
	TestPHDFilter(string _firstFrameFileName, string _groundTruthFileName, string model_cfg,
		string model_binary, string class_names, float min_confidence, int _npart);
	void run();
private:
	string firstFrameFileName, groundTruthFileName, preDetectionFile, 
		model_cfg, model_binary, class_names;
	float min_confidence;
	int npart;
	YOLODetector detector;
	ImageGenerator generator;
};

#endif