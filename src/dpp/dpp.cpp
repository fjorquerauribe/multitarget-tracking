#include "dpp.hpp"

#ifndef PARAMS
	const double ALPHA = 0.9;
	const double BETA = 1.1;
#endif

DPP::DPP(){}

vector<MyTarget> DPP::run(vector<MyTarget> raw_detections, VectorXd &weights, MatrixXd &features, double epsilon, double mu, double lambda)
{
	vector<MyTarget> results;

	if(raw_detections.size() > 0){
		VectorXd area(raw_detections.size());
		MatrixXd intersection(raw_detections.size(), raw_detections.size());

		for (size_t i = 0; i < raw_detections.size(); ++i)
		{
			Rect bbox = raw_detections.at(i).bbox;
			area(i) = bbox.width * bbox.height;
			for (size_t j = i; j < raw_detections.size(); ++j)
			{	
				Rect bbox2 = raw_detections.at(j).bbox;
				intersection(i,j) = intersection(j,i) = double((bbox & bbox2).area());
			}
		}

		MatrixXd sqrtArea = area.cwiseSqrt() * area.cwiseSqrt().adjoint();
		//MatrixXd rIntersectionArea = intersectionArea.array() / area.replicate(1, area.size()).adjoint().array();
		VectorXd nContain = VectorXd::Zero(raw_detections.size());

		/*for (unsigned int i = 0; i < rIntersectionArea.rows(); ++i)
		{
			for (unsigned int j = 0; j < rIntersectionArea.cols(); ++j)
			{
				if(rIntersectionArea(i,j) == 1)
					nContain(i) += 1;
			}
		}*/
		for (unsigned int i = 0; i < intersection.rows(); ++i)
		{
			for (unsigned int j = 0; j < intersection.cols(); ++j)
			{
				if( (intersection(i,j) / area(i)) == 1 )
					nContain(i) += 1;
			}
		}
		
		nContain = nContain.array() - 1;
		//cout << "nContain: " << nContain.transpose() << endl;
		
		VectorXd nPenalty = nContain.array().exp().pow(-lambda);
		VectorXd qualityTerm = weights;//getQualityTerm(weights, nPenalty);
		MatrixXd similarityTerm = getSimilarityTerm(features, intersection, sqrtArea, mu);
		vector<int> top = solve(qualityTerm, similarityTerm, epsilon);

		for (size_t i = 0; i < top.size(); ++i)
		{
			results.push_back(raw_detections.at(top.at(i)));
		}
	}
	
	return results;
}

vector<MyTarget> DPP::run(vector<MyTarget> tracks, double epsilon, double mu, double lambda)
{
	vector<MyTarget> results;

	if(tracks.size() > 0){
		VectorXd area(tracks.size());
		MatrixXd intersection(tracks.size(), tracks.size());
		MatrixXd features(tracks.size(), tracks.at(0).feature.size());
		VectorXd weights(tracks.size());

		for (size_t i = 0; i < tracks.size(); ++i)
		{
			Rect bbox = tracks.at(i).bbox;
			area(i) = bbox.width * bbox.height;
			features.row(i) = tracks.at(i).feature;
			weights(i) = tracks.at(i).score;

			for (size_t j = i; j < tracks.size(); ++j)
			{	
				Rect bbox2 = tracks.at(j).bbox;
				intersection(i,j) = intersection(j,i) = double((bbox & bbox2).area());
			}
		}

		MatrixXd sqrtArea = area.cwiseSqrt() * area.cwiseSqrt().adjoint();
		VectorXd nContain = VectorXd::Zero(tracks.size());

		for (unsigned int i = 0; i < intersection.rows(); ++i)
		{
			for (unsigned int j = 0; j < intersection.cols(); ++j)
			{
				if( (intersection(i,j) / area(i)) == 1 )
					nContain(i) += 1;
			}
		}

		nContain = nContain.array() - 1;
		
		VectorXd nPenalty = nContain.array().exp().pow(-lambda);
		VectorXd qualityTerm = getQualityTerm(weights, nPenalty);
		MatrixXd similarityTerm = getSimilarityTerm(features, intersection, sqrtArea, mu);
		vector<int> top = solve(qualityTerm, similarityTerm, epsilon);

		for (size_t i = 0; i < top.size(); ++i) results.push_back(tracks.at(top.at(i)));
	
	}
	
	return results;
}

vector<MyTarget> DPP::run(vector<MyTarget> tracks, double epsilon, Size img_size){
	vector<MyTarget> results;

	if(tracks.size() > 0){
		VectorXd quality_term(tracks.size());
		for (size_t i = 0; i < tracks.size(); ++i) quality_term(i) = sqrt(exp(tracks.at(i).score));
		MatrixXd similarity_term = affinity_kernel(tracks, img_size);
		vector<int> top = solve(quality_term, similarity_term, epsilon);
		for (size_t i = 0; i < top.size(); ++i) results.push_back(tracks.at(top.at(i)));	
	}
	return results;
}

VectorXd DPP::getQualityTerm(VectorXd &weights, VectorXd &nPenalty){
	/*** 
	 ***	Get quality term 
	 ***	qt = alpha * log(1 + s) + beta
	 ***/
	VectorXd qt = weights.cwiseProduct(nPenalty);
	qt = qt.array() / qt.maxCoeff();
	qt = qt.array() + 1;
	qt = qt.array().log() / log(10);
	qt = ALPHA * qt.array() + BETA;
	qt = qt.array().square();
	
	return qt;
}

MatrixXd DPP::getSimilarityTerm(MatrixXd &features, MatrixXd &intersection, MatrixXd &sqrtArea, double mu){
	/****
	 ****	Get similarity term
	 ****	S = w * Sc + (1 - w) * Ss
	 ****/

	MatrixXd Ss = intersection.array() / sqrtArea.array();
	MatrixXd Sc = features * features.adjoint();
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
			S_top.row(i) << similarityTerm.row(top.at(i));
		}

		for (unsigned int i = 0; i < remained.size(); ++i)
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

		double maxObj = prodQ * maxObj_;
		//cout << (maxObj / oldObj) << endl;
		
		if ( (maxObj / oldObj) > (epsilon) )
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

MatrixXd DPP::affinity_kernel(vector<MyTarget> tracks, Size img_size){
	MatrixXd kernel = MatrixXd(tracks.size(), tracks.size());
	double diag = sqrt( pow(img_size.height, 2) + pow(img_size.width, 2) );
	for(size_t i = 0; i < tracks.size(); i++){
		Rect box1 = tracks.at(i).bbox;

		for(size_t j = i; j < tracks.size(); j++){
			Rect box2 = tracks.at(j).bbox;
			double scale = exp( -100.0 * (sqrt(pow(box1.width - box2.width, 2)) + sqrt(pow(box1.height - box2.height, 2)))/diag );
			double pos = exp( -100.0 * (sqrt(pow(box1.x - box2.x, 2)) + sqrt(pow(box1.y - box2.y, 2)))/diag );

			kernel(i,j) = kernel(j,i) = scale * pos;
		}
	}
	//exit (EXIT_FAILURE);
	return kernel;
}

MatrixXd DPP::squared_exponential_kernel(MatrixXd X, double nu, double sigma_f){
    MatrixXd cov;
    int Xnrow = X.rows();
    int Xncol = X.cols();
    cov.resize(Xnrow, Xnrow);
    for(int i = 0; i < (Xnrow-1); ++i){
      for(int j = i + 1; j < Xnrow; ++j){
        double S = 0;
        for(int k = 0; k < Xncol; ++k){
          double d = ( X(i,k) - X(j,k));
          S += exp(nu) * d * d;
        }
        cov(i,j) = exp(sigma_f - 0.5 * S);
        cov(j,i) = cov(i,j);
      }
    }
    for(int i = 0; i < Xnrow; ++i){
        cov(i,i) =  exp(sigma_f);
    }
    return cov;
}