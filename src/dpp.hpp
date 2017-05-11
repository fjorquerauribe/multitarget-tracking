#ifndef DPP_H
#define DPP_H

#include <stdlib.h>
#include <cstdlib>
#include <opencv2/imgproc.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

using namespace cv;
using namespace std;
using namespace Eigen;

class DPP
{
public:
	DPP();
	vector<Rect> run(vector<Rect> preDetections, VectorXd &detectionWeights, MatrixXd &featureValues, double epsilon = 0.1, double mu = 0.7, double lambda = 0.1);

private:
	VectorXd getQualityTerm(VectorXd &detectionWeights, VectorXd &nPenalty);
	MatrixXd getSimilarityTerm(MatrixXd &featureValues, MatrixXd &intersectionArea, MatrixXd &sqrtArea, double mu);
	vector<int> solve(VectorXd &qualityTerm, MatrixXd &similarityTerm, double epsilon);
};

#endif