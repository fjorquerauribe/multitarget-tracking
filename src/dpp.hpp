#ifndef DPP_H
#define DPP_H

#include <stdlib.h>
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
	vector<Rect> run(vector<Rect> preDetections, VectorXd &detectionWeights, MatrixXd &featureValues);

private:
	VectorXd getQualityTerm(VectorXd &detectionWeights, VectorXd &nPenalty);
	MatrixXd getSimilarityTerm(MatrixXd &featureValues, MatrixXd &intersectionArea, MatrixXd &sqrtArea);
	vector<int> solve(VectorXd &qualityTerm, MatrixXd &similarityTerm);
};

#endif