#include "dpp.hpp"

DPP::DPP(){}

void DPP::run(vector<Rect> preDetections, VectorXd &detectionWeights, MatrixXd &featureValues, double alpha, double lambda, double beta, double mu, double epsilon)
{
	VectorXd area(preDetections.size());
	MatrixXd intersectionArea(preDetections.size(), preDetections.size());
	
	//cout << "preDetections size: " << preDetections.size() << endl;
	for (size_t i = 0; i < preDetections.size(); ++i)
	{
		Rect bbox = preDetections.at(i);
		area(i) = bbox.width * bbox.height;
		
		for (size_t j = 0; j < preDetections.size(); ++j)
		{	Rect bbox2 = preDetections.at(j);
			intersectionArea(i,j) = double((bbox & bbox2).area());
		}
			
	}
	
	MatrixXd sqrtArea = area.cwiseSqrt() * area.cwiseSqrt().adjoint();	
	MatrixXd rIntersectionArea = intersectionArea.array() / area.replicate(1, area.size()).adjoint().array();

	VectorXd nContain = VectorXd::Zero(rIntersectionArea.rows());
	for (int i = 0; i < rIntersectionArea.rows(); ++i)
	{
		for (int j = 0; j < rIntersectionArea.cols(); ++j)
		{
			if(rIntersectionArea(i,j) == 1)
				nContain(i) += 1;
		}
	}
	nContain = nContain.array() - 1;

	VectorXd nPenalty = nContain.array().exp().pow(lambda);

	VectorXd qualityTerm = getQualityTerm(detectionWeights, nPenalty, alpha, beta);
	cout << "qualityTerm: " << endl;
	cout << qualityTerm << endl;

	MatrixXd similarityTerm = getSimilarityTerm(featureValues, intersectionArea, sqrtArea, mu);
	cout << "similarityTerm: " << endl;
	cout << similarityTerm << endl;
	//cout << "nPenalty:" << endl;
	//cout << nPenalty << endl;
}

VectorXd DPP::getQualityTerm(VectorXd &detectionWeights, VectorXd &nPenalty, double alpha, double beta){
	VectorXd qt = detectionWeights.cwiseProduct(nPenalty);
	//double maxQt = qt.maxCoeff();
	qt = qt.array() / qt.maxCoeff();
	qt = qt.array() + 1;
	qt = qt.array().log() / log(10);
	qt = alpha * qt.array() + beta;
	qt = qt.array().square();
	return qt;
}

MatrixXd DPP::getSimilarityTerm(MatrixXd &featureValues, MatrixXd &intersectionArea, MatrixXd &sqrtArea, double mu){
	MatrixXd Ss = intersectionArea.array() / sqrtArea.array();
	MatrixXd Sc = featureValues.array() * featureValues.adjoint().array();
	MatrixXd S = mu * Ss.array() + (1 - mu) * Sc.array();
	return S;
}