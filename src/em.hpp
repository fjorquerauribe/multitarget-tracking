#ifndef EM_H
#define EM_H

#include <iostream>
#include <stdlib.h>
#include <cfloat>
#include <cmath>
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Cholesky>
#include "likelihood/multivariate_gaussian.hpp"

using namespace Eigen;
using namespace std;

class EM {
	public:
		EM(MatrixXd &data,int n_components,bool _diag=true);
		double estep();
		void mstep();
		double fit(int n_iter);
		vector<VectorXd> getMeans();	
	private:
		VectorXd random_generator(int dim);
		double random_uniform();
		int rows, components, dim;
		double loglike;
		vector<VectorXd> means;
		vector<MatrixXd> covs;
		vector<double> pi;
		MatrixXd data, resp;
		VectorXd data_mean;
		MatrixXd data_cov;
		bool diag;
};


#endif // EM