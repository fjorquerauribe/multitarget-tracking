#include "dpp.hpp"

#ifndef PARAMS
const double ALPHA = 0.9; //0.9
//const double LAMBDA = 0.1;
const double BETA = 1.1;
//const double MU = 0.7;
//const double EPSILON = 0.1;
#endif

DPP::DPP(){}

vector<Rect> DPP::run(vector<Rect> preDetections, VectorXd &detectionWeights, MatrixXd &featureValues, double epsilon, double mu, double lambda)
{
	VectorXd area(preDetections.size());
	MatrixXd intersectionArea(preDetections.size(), preDetections.size());

	//cout << "preDetections size: " << preDetections.size() << endl;
	for (size_t i = 0; i < preDetections.size(); ++i)
	{
		Rect bbox = preDetections.at(i);
		area(i) = bbox.width * bbox.height;
		for (size_t j = i; j < preDetections.size(); ++j)
		{	
			Rect bbox2 = preDetections.at(j);
			intersectionArea(i,j) = intersectionArea(j,i) = double((bbox & bbox2).area());
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
	
	VectorXd qualityTerm = getQualityTerm(detectionWeights, nPenalty);
	MatrixXd similarityTerm = getSimilarityTerm(featureValues, intersectionArea, sqrtArea, mu);
	vector<int> top = solve(qualityTerm, similarityTerm, epsilon);

	/*MatrixXd similarityTerm = getSimilarityTerm(featureValues, intersectionArea, sqrtArea);
	vector<int> top = solve(detectionWeights, similarityTerm);*/

	vector<Rect> respDPP;
	for (int i = 0; i < top.size(); ++i)
	{
		respDPP.push_back(preDetections.at(top.at(i)));
	}
	
	return respDPP;
}

VectorXd DPP::getQualityTerm(VectorXd &detectionWeights, VectorXd &nPenalty){
	/*** 
	 ***	Get quality term 
	 ***	qt = alpha * log(1 + s) + beta
	 ***/

	VectorXd qt = detectionWeights.cwiseProduct(nPenalty);
	qt = qt.array() / qt.maxCoeff();
	qt = qt.array() + 1;
	qt = qt.array().log() / log(10);
	qt = ALPHA * qt.array() + BETA;
	qt = qt.array().square();
	
	return qt;
}

MatrixXd DPP::getSimilarityTerm(MatrixXd &featureValues, MatrixXd &intersectionArea, MatrixXd &sqrtArea, double mu){
	/****
	 ****	Get similarity term
	 ****	S = w * Sc + (1 - w) * Ss
	 ****/

	MatrixXd Ss = intersectionArea.array() / sqrtArea.array();
	MatrixXd Sc = featureValues * featureValues.adjoint();
	MatrixXd S = mu * Ss.array() + (1 - mu) * Sc.array();
	return S;
}

vector<int> DPP::solve(VectorXd &qualityTerm, MatrixXd &similarityTerm, double epsilon){
	VectorXi remained = VectorXi::LinSpaced(qualityTerm.size(), 0, qualityTerm.size() - 1);
	int selected;
	double oldObj, prodQ;
	oldObj = qualityTerm.maxCoeff(&selected);
	
	vector<int> top;
	top.push_back(selected);
	MatrixXd oldS = MatrixXd::Identity(1,1);
	prodQ = oldObj;

	/*cout << "qualityTerm:" << endl;
	cout << qualityTerm.transpose() << endl;
	cout << "similarityTerm" << endl;
	cout << similarityTerm << endl;
	exit(EXIT_FAILURE);*/

	while(true){
		double maxObj_ = 0;
		copy( remained.data() + selected + 1, remained.data() + remained.size(), remained.data() + selected ); // delete selected item
		remained.conservativeResize(remained.size() - 1);

		MatrixXd newS = MatrixXd::Identity( oldS.rows() + 1, oldS.cols() + 1 );
		MatrixXd maxS( oldS.rows() + 1, oldS.cols() + 1 );
		newS.block(0, 0, oldS.rows(), oldS.cols()) << oldS;

		MatrixXd S_top(top.size(), similarityTerm.cols());
		for (size_t i = 0; i < top.size(); ++i)
		{
			//S_top.row(i) << similarityTerm.row(i);
			S_top.row(i) << similarityTerm.row(top.at(i));
		}

		for (int i = 0; i < remained.size(); ++i)
		{
			VectorXd tmp = S_top.col(remained(i));

			newS.block(0, newS.cols() - 1, newS.rows() - 1, 1) << tmp;
			newS.block(newS.rows() - 1, 0, 1, newS.cols() - 1) << tmp.transpose();
			double obj_ = qualityTerm(remained(i)) * newS.determinant();
			
			if (obj_ > maxObj_)
			{
				selected = i;
				maxObj_ = obj_;
				maxS = newS;
			}
		}

		double maxObj = prodQ * maxObj_ ;
		cout << "maxObj/oldObj:" << (maxObj / oldObj) << ", epsilon:" << (1 + epsilon) << endl;
		if ( (maxObj / oldObj) > (1 + epsilon) )
		//if ( (maxObj / oldObj) > epsilon )
		{
			top.push_back(remained(selected));
			oldObj = maxObj;
			oldS = maxS;
			prodQ = prodQ * qualityTerm(remained(selected));
		}
		else{
			break;
		}
	}

	return top;
}




