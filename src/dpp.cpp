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

	//cout << "qt" << endl;
	VectorXd qualityTerm = getQualityTerm(detectionWeights, nPenalty, alpha, beta);
	
	//cout << "st" << endl;
	MatrixXd similarityTerm = getSimilarityTerm(featureValues, intersectionArea, sqrtArea, mu);
	
	//cout << "solve" << endl;
	solve(qualityTerm, similarityTerm, epsilon);
	
	//cout << "end" << endl;
	//exit( EXIT_FAILURE );
}

VectorXd DPP::getQualityTerm(VectorXd &detectionWeights, VectorXd &nPenalty, double alpha, double beta){
	/*** 
	 ***	Get quality term 
	 ***	q = alpha * s + beta
	 ***/

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
	/****
	 ****	Get similarity term
	 ****	S = w * S^c + (1 - w) * S^s
	 ****/

	MatrixXd Ss = intersectionArea.array() / sqrtArea.array();
	MatrixXd Sc = featureValues * featureValues.adjoint();
	MatrixXd S = mu * Ss.array() + (1 - mu) * Sc.array();
	return S;
}

void DPP::solve(VectorXd &qualityTerm, MatrixXd &similarityTerm, double epsilon){
	VectorXd remained = VectorXd::LinSpaced(qualityTerm.size(), 1, qualityTerm.size());
	int oldObj, selected, top, prodQ;
	oldObj = qualityTerm.maxCoeff(&selected);
	top = selected;
	MatrixXd oldS(1,1); oldS << 1;
	prodQ = oldObj;

	while(true){
		int maxObj_ = 0;
		copy( remained.data() + selected + 1, remained.data() + remained.size(), remained.data() + selected ); // delete select item
		MatrixXd newS = MatrixXd::Identity( oldS.rows() + 1, oldS.cols() + 1 );
		newS.block(0,0, oldS.rows(), oldS.cols()) << oldS;
		VectorXd S_top(similarityTerm.row(top).size());
		S_top << similarityTerm.row(top);

		break;

		for (int i = 0; i < remained.size(); ++i)
		{
			/* code */
		}
	}
}




