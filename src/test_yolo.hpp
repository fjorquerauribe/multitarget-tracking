#ifndef TEST_YOLO_H
#define TEST_YOLO_H

#include <time.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <opencv2/opencv.hpp>
#include "utils/image_generator.hpp"
#include "utils/utils.hpp"
#include "models/phd_particle_filter.hpp"
#include "detectors/yolo_detector.hpp"

using namespace std;
using namespace cv;

class TestYOLODetector{
    public:
        TestYOLODetector();
        TestYOLODetector(string first_frame_file, string ground_truth_filename, string model_cfg, string model_binary, string class_names, float min_confidence); 
        void run(bool verbose);
    private:
        string first_frame_file, ground_truth_filename, model_cfg, model_binary, class_names;
        float min_confidence;
        ImageGenerator generator;
};

#endif