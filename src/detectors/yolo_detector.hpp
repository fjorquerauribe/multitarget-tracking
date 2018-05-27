#ifndef YOLO_DETECTOR_H
#define YOLO_DETECTOR_H

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <Eigen/Core>

using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace Eigen;

class YOLODetector{
    public:
        YOLODetector(string model_cfg, string model_binary, string class_names, float min_confidence);
        vector<Rect> detect(Mat frame);
        void draw(Mat frame, Scalar color = Scalar(0, 255, 0));

    private:
        vector<Rect> detections;
        VectorXd weights;
        vector<String> classes;
        Net net;
        float min_confidence;
        vector<String> class_names_vec;
};

#endif