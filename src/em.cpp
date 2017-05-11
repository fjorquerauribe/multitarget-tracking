#include "em.hpp"

EM::EM(MatrixXd &_data, int n_components, bool _diag){
	MVNGaussian element(_data);
	this->components = n_components; 
	this->rows = _data.rows();							// No. of data
	this->dim = _data.cols();
	this->data = _data;
	this->data_mean = element.getMean();
	this->resp = MatrixXd::Zero(this->rows, this->components);
	this->covs.reserve(this->components);
	this->means.reserve(this->components);
	this->pi.reserve(this->components);
	this->diag = _diag;
	if(this->diag){
		this->data_cov = 10 * MatrixXd::Identity(this->dim, this->dim);
	}
	else{
		this->data_cov = element.getCov();
	}
	for(int i = 0; i < this->components; i++){
		this->pi.push_back(1.0f/this->components);				// Uniform prior of p(k)
		this->covs.push_back(this->data_cov);
		//this->means.push_back(this->data_mean + 10 * VectorXd::Random(this->dim));

		VectorXd randVec = random_generator(this->dim);
		this->means.push_back(this->data_mean + 10 * randVec);
	}
}

double EM::estep(){
	this->resp = MatrixXd::Zero(this->rows, this->components);
	this->loglike = 0;
	for(int n = 0; n < this->rows; n++) {
		for (int i = 0; i < this->components; i++){
			double logdet = log(this->covs[i].determinant());
			if (!isfinite(logdet)) {
				this->covs[i] = this->data_cov;
				break;
			}
			if (this->means[i].hasNaN()) {
				//this->means[i] = data_mean + 10 * VectorXd::Random(this->dim);
				VectorXd randVec = random_generator(this->dim);
				this->means[i] = this->data_mean + 10 * randVec;
				break;
			}
			LLT<MatrixXd> chol(this->covs[i]);
			MatrixXd L = chol.matrixL();
			MatrixXd cov_inverse = L.adjoint().inverse() * L.inverse();
			VectorXd tmp1 = this->data.row(n);
	        tmp1 -= this->means[i];
	        MatrixXd tmp2 = tmp1.transpose() * cov_inverse;
	        tmp2 = tmp2 * tmp1;
	        this->resp(n,i) = log(this->pi[i]) - 0.5 * tmp2(0,0) - (this->dim/2.0f) * log(2.0f * M_PI) -(0.5) * logdet;
    	}
    	double sumexp = 0.0;
    	double max_value = this->resp.row(n).maxCoeff();
        for (int i = 0; i < this->components; i++) {
        	sumexp += exp(this->resp(n,i) - max_value);
        }
        double norm_const = max_value + log(sumexp);
        this->resp.row(n) = (this->resp.row(n).array() - norm_const).exp().matrix();
        this->loglike += norm_const;
	}
	return this->loglike;
}

vector<VectorXd> EM::getMeans(){
	return this->means;
}

void EM::mstep(){
	//double wgt,sum;
	for(int i = 0; i < this->components; i++){
		double wgt = 0.0;
		VectorXd vec_sum = RowVectorXd::Zero(this->dim);
		for(int n = 0; n < this->rows; n++){
			wgt += resp(n,i);
			vec_sum += resp(n,i) * this->data.row(n).transpose();
		}
		this->pi[i] = wgt/this->rows;
		this->means[i] = vec_sum;
		this->means[i] /= wgt;
		if(!this->diag){
			for(int m = 0; m < this->dim; m++){
				double sum = 0.0;
				for(int j = 0; j < this->dim; j++){
					for(int n = 0; n < this->rows; n++){
						sum += resp(n,i) * (this->data(n,m) - this->means[i](m)) * (this->data(n,j) - this->means[i](j));
					}
					this->covs[i](m,j) = sum/wgt;
				}
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

VectorXd EM::random_generator(int dimension){
	unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  	mt19937 generator;
  	generator.seed(seed1);
  	normal_distribution<double> dnormal(0.0,1.0);
	VectorXd random_vector(dimension);

	for (int i = 0; i < dimension; ++i){
		random_vector(i) = dnormal(generator);
	}
	return random_vector;
}

double EM::random_uniform(){
	random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 1);
    return dis(gen);
}