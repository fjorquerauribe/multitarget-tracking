#ifndef CPU_LR_HOG_DETECTOR_H
#define CPU_LR_HOG_DETECTOR_H
#include <Eigen/Dense>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/ximgproc/segmentation.hpp>
#include <iostream>
#include <random>
#include <chrono>
#include <string>
#include "../utils/c_utils.hpp"
#include "../libs/nms/nms.hpp"
#include "../dpp/dpp.hpp"
#include "../likelihood/logistic_regression.hpp"
#include "../likelihood/CPU_logistic_regression.hpp"
#include "../libs/piotr_fhog/fhog.hpp"


struct Args {
	bool make_gray;
    bool resize_src;
    int hog_width;
    int hog_height;
    double gr_threshold;
    double hit_threshold;
    int n_orients;
    int bin_size;
    double overlap_threshold;
    double p_accept;
    double lambda;
    double epsilon;
    double tolerance;
    int n_iterations;
} ;

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace cv::ximgproc::segmentation;
class CPU_LR_HOGDetector 
{
public:
	void init(double group_threshold, double hit_threshold);
	vector<Rect> detect(Mat &frame, vector<Rect> &detections, vector<double> &weights, MatrixXd &features);
	void train();
	VectorXd getFeatures(Mat &frame);
    vector<double> getWeights();
	void generateFeatures(Mat &frame, double label);
	void dataClean();
	void draw();
	void saveToCSV(string name, bool append = true);
	void loadModel(VectorXd weights,VectorXd featureMean, VectorXd featureStd, VectorXd featureMax, VectorXd featureMin, double bias);
	void loadFeatures(MatrixXd features, VectorXd labels);
	VectorXd predictTest(MatrixXd features,bool data_processing);
	
protected:
	Args args;
	VectorXd genHog(Mat &frame);
	VectorXd genRawPixels(Mat &frame);
	HOGDescriptor hog;
	CPU_LogisticRegression logistic_regression;
	int num_frame=0;
	double max_value=1.0;
	MatrixXd feature_values;
	int group_threshold;
	double hit_threshold;
	int n_descriptors, n_data;
	vector<Rect> detections;
	VectorXd labels;
	vector<double> weights;
	Mat frame;
	C_utils tools;
	mt19937 generator;
	bool initialized;

};

#endif
