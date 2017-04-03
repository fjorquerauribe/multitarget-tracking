#include "em.hpp"

EM::EM(MatrixXd &_data, int n_components){
	MVNGaussian element(_data);
	components = n_components; 
	rows = _data.rows();							// No. of data
	dim = _data.cols();
	data = _data;
	data_mean = element.getMean();
	resp = MatrixXd::Zero(rows,components);
	covs.reserve(components);
	means.reserve(components);
	pi.reserve(components);
	data_cov = element.getCov();
	for(int i = 0; i < components; i++){
		pi.push_back(1./components);				// Uniform prior of p(k)
		covs.push_back(data_cov);
		means.push_back(data_mean + 100 * VectorXd::Random(dim));
	}
}

double EM::estep(){
	resp = MatrixXd::Zero(rows, components);
	loglike = 0;
	for(int n = 0; n < rows; n++) {
		for (int i = 0; i < components; i++){
			double logdet = log(covs[i].determinant());
			if (!isfinite(logdet)) {
				covs[i] = data_cov;
				break;
			}
			if (means[i].hasNaN()) {
				means[i] = data_mean + 10 * VectorXd::Random(dim);
				break;
			}
			LLT<MatrixXd> chol(covs[i]);
			MatrixXd L = chol.matrixL();
			MatrixXd cov_inverse = L.adjoint().inverse() * L.inverse();
			VectorXd tmp1 = data.row(n);
	        tmp1 -= means[i];
	        MatrixXd tmp2 = tmp1.transpose() * cov_inverse;
	        tmp2 = tmp2 * tmp1;
	        resp(n,i) = log(pi[i]) -0.5 * tmp2(0,0) - (dim/2) * log(2*M_PI) -(0.5) * logdet;
    	}
    	double sumexp=0.0;
    	double max_value = resp.row(n).maxCoeff();
        for (int i = 0; i < components; i++) {
        	sumexp += exp(resp(n,i) - max_value);
        }
        double norm_const = max_value + log(sumexp);
        resp.row(n) = (resp.row(n).array() - norm_const).exp().matrix();
        loglike += norm_const;
	}
	return loglike;
}

vector<VectorXd> EM::getMeans(){
	return means;
}

void EM::mstep(){
	//double wgt,sum;
	for(int i = 0; i < components; i++){
		double wgt = 0.0;
		VectorXd vec_sum = RowVectorXd::Zero(dim);
		for(int n = 0; n < rows; n++){
			wgt += resp(n,i);
			vec_sum += resp(n,i) * data.row(n).transpose();
		}
		pi[i] = wgt/rows;
		means[i] = vec_sum;
		means[i] /= wgt;
		//MatrixXd centered = data.rowwise() - means[i].transpose();
		//for(int n=0;n<centered.rows();n++){
		//	centered.row(n)*=resp(n,i);
		//}
    	//covs[i] = (centered.adjoint() * centered) / wgt;
		for(int m = 0; m < dim; m++){
			double sum = 0.0;
			for(int j = 0; j < dim; j++){
				for(int n = 0; n < rows; n++){
					sum += resp(n,i) * (data(n,m) - means[i](m)) * (data(n,j) - means[i](j));
				}
				covs[i](m,j) = sum/wgt;
			}
		}
	}
}

double EM::fit(int n_iter){
	double result = 0.;
	double oldresult;
	for(int i = 0; i < n_iter; i++){
		oldresult = result;
		result = estep();
		mstep();
		//cout << "log-like:" << result << endl;
	}
	return result - oldresult;
}

/*int main(){
	//MatrixXd m = 100 * MatrixXd::Random(100,4);
	MatrixXd m(150,4);
	m << 5.1,3.5,1.4,0.2,
	4.9,3.0,1.4,0.2,
	4.7,3.2,1.3,0.2,
	4.6,3.1,1.5,0.2,
	5.0,3.6,1.4,0.2,
	5.4,3.9,1.7,0.4,
	4.6,3.4,1.4,0.3,
	5.0,3.4,1.5,0.2,
	4.4,2.9,1.4,0.2,
	4.9,3.1,1.5,0.1,
	5.4,3.7,1.5,0.2,
	4.8,3.4,1.6,0.2,
	4.8,3.0,1.4,0.1,
	4.3,3.0,1.1,0.1,
	5.8,4.0,1.2,0.2,
	5.7,4.4,1.5,0.4,
	5.4,3.9,1.3,0.4,
	5.1,3.5,1.4,0.3,
	5.7,3.8,1.7,0.3,
	5.1,3.8,1.5,0.3,
	5.4,3.4,1.7,0.2,
	5.1,3.7,1.5,0.4,
	4.6,3.6,1.0,0.2,
	5.1,3.3,1.7,0.5,
	4.8,3.4,1.9,0.2,
	5.0,3.0,1.6,0.2,
	5.0,3.4,1.6,0.4,
	5.2,3.5,1.5,0.2,
	5.2,3.4,1.4,0.2,
	4.7,3.2,1.6,0.2,
	4.8,3.1,1.6,0.2,
	5.4,3.4,1.5,0.4,
	5.2,4.1,1.5,0.1,
	5.5,4.2,1.4,0.2,
	4.9,3.1,1.5,0.1,
	5.0,3.2,1.2,0.2,
	5.5,3.5,1.3,0.2,
	4.9,3.1,1.5,0.1,
	4.4,3.0,1.3,0.2,
	5.1,3.4,1.5,0.2,
	5.0,3.5,1.3,0.3,
	4.5,2.3,1.3,0.3,
	4.4,3.2,1.3,0.2,
	5.0,3.5,1.6,0.6,
	5.1,3.8,1.9,0.4,
	4.8,3.0,1.4,0.3,
	5.1,3.8,1.6,0.2,
	4.6,3.2,1.4,0.2,
	5.3,3.7,1.5,0.2,
	5.0,3.3,1.4,0.2,
	7.0,3.2,4.7,1.4,
	6.4,3.2,4.5,1.5,
	6.9,3.1,4.9,1.5,
	5.5,2.3,4.0,1.3,
	6.5,2.8,4.6,1.5,
	5.7,2.8,4.5,1.3,
	6.3,3.3,4.7,1.6,
	4.9,2.4,3.3,1.0,
	6.6,2.9,4.6,1.3,
	5.2,2.7,3.9,1.4,
	5.0,2.0,3.5,1.0,
	5.9,3.0,4.2,1.5,
	6.0,2.2,4.0,1.0,
	6.1,2.9,4.7,1.4,
	5.6,2.9,3.6,1.3,
	6.7,3.1,4.4,1.4,
	5.6,3.0,4.5,1.5,
	5.8,2.7,4.1,1.0,
	6.2,2.2,4.5,1.5,
	5.6,2.5,3.9,1.1,
	5.9,3.2,4.8,1.8,
	6.1,2.8,4.0,1.3,
	6.3,2.5,4.9,1.5,
	6.1,2.8,4.7,1.2,
	6.4,2.9,4.3,1.3,
	6.6,3.0,4.4,1.4,
	6.8,2.8,4.8,1.4,
	6.7,3.0,5.0,1.7,
	6.0,2.9,4.5,1.5,
	5.7,2.6,3.5,1.0,
	5.5,2.4,3.8,1.1,
	5.5,2.4,3.7,1.0,
	5.8,2.7,3.9,1.2,
	6.0,2.7,5.1,1.6,
	5.4,3.0,4.5,1.5,
	6.0,3.4,4.5,1.6,
	6.7,3.1,4.7,1.5,
	6.3,2.3,4.4,1.3,
	5.6,3.0,4.1,1.3,
	5.5,2.5,4.0,1.3,
	5.5,2.6,4.4,1.2,
	6.1,3.0,4.6,1.4,
	5.8,2.6,4.0,1.2,
	5.0,2.3,3.3,1.0,
	5.6,2.7,4.2,1.3,
	5.7,3.0,4.2,1.2,
	5.7,2.9,4.2,1.3,
	6.2,2.9,4.3,1.3,
	5.1,2.5,3.0,1.1,
	5.7,2.8,4.1,1.3,
	6.3,3.3,6.0,2.5,
	5.8,2.7,5.1,1.9,
	7.1,3.0,5.9,2.1,
	6.3,2.9,5.6,1.8,
	6.5,3.0,5.8,2.2,
	7.6,3.0,6.6,2.1,
	4.9,2.5,4.5,1.7,
	7.3,2.9,6.3,1.8,
	6.7,2.5,5.8,1.8,
	7.2,3.6,6.1,2.5,
	6.5,3.2,5.1,2.0,
	6.4,2.7,5.3,1.9,
	6.8,3.0,5.5,2.1,
	5.7,2.5,5.0,2.0,
	5.8,2.8,5.1,2.4,
	6.4,3.2,5.3,2.3,
	6.5,3.0,5.5,1.8,
	7.7,3.8,6.7,2.2,
	7.7,2.6,6.9,2.3,
	6.0,2.2,5.0,1.5,
	6.9,3.2,5.7,2.3,
	5.6,2.8,4.9,2.0,
	7.7,2.8,6.7,2.0,
	6.3,2.7,4.9,1.8,
	6.7,3.3,5.7,2.1,
	7.2,3.2,6.0,1.8,
	6.2,2.8,4.8,1.8,
	6.1,3.0,4.9,1.8,
	6.4,2.8,5.6,2.1,
	7.2,3.0,5.8,1.6,
	7.4,2.8,6.1,1.9,
	7.9,3.8,6.4,2.0,
	6.4,2.8,5.6,2.2,
	6.3,2.8,5.1,1.5,
	6.1,2.6,5.6,1.4,
	7.7,3.0,6.1,2.3,
	6.3,3.4,5.6,2.4,
	6.4,3.1,5.5,1.8,
	6.0,3.0,4.8,1.8,
	6.9,3.1,5.4,2.1,
	6.7,3.1,5.6,2.4,
	6.9,3.1,5.1,2.3,
	5.8,2.7,5.1,1.9,
	6.8,3.2,5.9,2.3,
	6.7,3.3,5.7,2.5,
	6.7,3.0,5.2,2.3,
	6.3,2.5,5.0,1.9,
	6.5,3.0,5.2,2.0,
	6.2,3.4,5.4,2.3,
	5.9,3.0,5.1,1.8;
	EM element(m,4);
	cout << element.fit(100) << endl;
	return 0;
}
*/