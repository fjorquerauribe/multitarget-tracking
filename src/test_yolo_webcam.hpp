#ifndef TEST_YOLO_H
#define TEST_YOLO_H

#include <time.h>
#include <iostream>
#include <cstdlib>
#include <queue>
#include <thread> 
#include <opencv2/opencv.hpp>
#include "utils/utils.hpp"
#include "models/phd_gaussian_mixture.hpp"
#include "detectors/yolo_detector.hpp"

using namespace std;
using namespace cv;

class TestYOLOWebcam{
    public:
        TestYOLOWebcam();
        TestYOLOWebcam(string model_cfg, string model_binary, string class_names, float min_confidence,int video); 
        void run(bool verbose,queue<Mat> &frame_buffer);
        void video_capture_buffer(queue<Mat> &frame_buffer);

    private:
        string model_cfg, model_binary, class_names;
        float min_confidence;
        VideoCapture cap;
};

#endif