#ifndef DPP_H
#define DPP_H

#include <stdlib.h>
#include <cstdlib>
#include <opencv2/imgproc.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include "utils.hpp"

using namespace cv;
using namespace std;
using namespace Eigen;

class DPP
{
public:
	DPP();
	vector<MyTarget> run(vector<MyTarget> raw_detections, VectorXd &weights, MatrixXd &features, double epsilon = 0.1, double mu = 0.7, double lambda = 0.1);
	vector<MyTarget> run(vector<MyTarget> tracks, double epsilon, double mu, double lambda);
	vector<MyTarget> run(vector<MyTarget> tracks, double epsilon, Size img_size);
	//vector<MyTarget> run(vector<MyTarget> tracks, double epsilon);
private:
	VectorXd getQualityTerm(VectorXd &weights, VectorXd &nPenalty);
	MatrixXd getSimilarityTerm(MatrixXd &features, MatrixXd &intersection, MatrixXd &sqrtArea, double mu);
	MatrixXd affinity_kernel(vector<MyTarget> tracks, Size img_size);
	MatrixXd squared_exponential_kernel(MatrixXd X, double nu, double sigma_f);
	vector<int> solve(VectorXd &qualityTerm, MatrixXd &similarityTerm, double epsilon);
};

#endif