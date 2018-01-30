#include "test_gm_phd.hpp"

TestGMPHDFilter::TestGMPHDFilter(){}

TestGMPHDFilter::TestGMPHDFilter(string _firstFrameFileName, 
	string _groundTruthFileName, string _detectionFile)
{
	this->firstFrameFileName = _firstFrameFileName;
	this->groundTruthFileName = _groundTruthFileName;
	this->detectionFile = _detectionFile;
	this->generator = ImageGenerator(this->firstFrameFileName, this->groundTruthFileName, this->detectionFile);
	//this->pruningMethod = _pruningMethod;
}

void TestGMPHDFilter::run(bool verbose){
	PHDGaussianMixture filter(verbose);
	run(verbose, filter);
}

void TestGMPHDFilter::run(bool verbose, double epsilon){
	PHDGaussianMixture filter(verbose, epsilon);
	run(verbose, filter);
}

void TestGMPHDFilter::run(bool verbose, double threshold, int neighbors, double min_scores_num){
	PHDGaussianMixture filter(verbose, threshold, neighbors, min_scores_num);
	run(verbose, filter);
}

void TestGMPHDFilter::run(bool verbose, PHDGaussianMixture filter)
{
	RNG rng( 0xFFFFFFFF );
	map<int,Scalar> color;
	//PHDGaussianMixture filter(verbose);
	if(verbose) namedWindow("PHD Gaussian Mixture", WINDOW_NORMAL);
	
	for (size_t i = 0; i < this->generator.getDatasetSize(); ++i)
	{
		Mat currentFrame = this->generator.getFrame(i);
		vector<Target> gt = this->generator.getGroundTruth(i);
		vector<Rect> detections = this->generator.getDetections(i);
		MatrixXd features = this->generator.getDetectionFeatures(i);
		VectorXd weights = this->generator.getDetectionWeights(i);
		
		vector<Target> estimates;
		
		if(verbose)	cout << "Target number: " << gt.size() << endl;

		if (!filter.is_initialized() &&  detections.size()>0)
		{
			filter.initialize(currentFrame, detections, features, weights);
			estimates = filter.estimate(currentFrame, true);
		}
		else
		{
			filter.predict();
			filter.update(currentFrame, detections, features, weights);
			estimates = filter.estimate(currentFrame, true);
		}

		for(size_t j = 0; j < estimates.size(); j++){
                cout << i + 1
                << "," << estimates.at(j).label
                << "," << estimates.at(j).bbox.x
                << "," << estimates.at(j).bbox.y
                << "," << estimates.at(j).bbox.width
                << "," << estimates.at(j).bbox.height
                << ",1,-1,-1,-1" << endl;
        }

		if(verbose) {
            cout << "----------------------------------------" << endl;
			imshow("PHD Gaussian Mixture", currentFrame);
			//imwrite("gm_phd_" + to_string(i) + ".png", currentFrame);
			waitKey(100);
		}
	}
}

int main(int argc, char const *argv[])
{
	string _firstFrameFileName, _gtFileName, _detectionFile;
	bool verbose = false;
	double dpp_epsilon, nms_threshold, nms_min_scores_num;
	int nms_neighbors;
	if((argc != 9) && (argc != 13) && (argc != 15) && (argc != 17))
	{
		cout << "Incorrect input list" << endl;
		cout << "exiting..." << endl;
		return EXIT_FAILURE;
	}
	else
	{
		if(strcmp(argv[1], "-img") == 0)
	  	{
	    	_firstFrameFileName = argv[2];
	  	}
	  	else
	  	{
	  		cout << "No images given" << endl;
	  		cout << "exiting..." << endl;
	  		return EXIT_FAILURE;
	  	}
	  	if(strcmp(argv[3], "-gt") == 0)
	  	{
	    	_gtFileName = argv[4];
	  	}
	  	else
	  	{
	  		cout << "No ground truth given" << endl;
	  		cout << "exiting..." << endl;
	  		return EXIT_FAILURE;
	  	}
	  	if(strcmp(argv[5], "-det") == 0)
	  	{
	    	_detectionFile = argv[6];
	  	}
	  	else
	  	{
	  		cout << "No detections file given" << endl;
	  		cout << "exiting..." << endl;
	  		return EXIT_FAILURE;
		}
		if(strcmp(argv[7], "-pruning") == 0){
			if(strcmp(argv[8], "dpp") == 0){
				if(strcmp(argv[9], "-epsilon") == 0){
					dpp_epsilon = stod(argv[10]);
				}
				else{
					cout << "No dpp epsilon value given" << endl;
					cout << "exiting..." << endl;
					return EXIT_FAILURE;
				}
				if(strcmp(argv[11], "-verbose") == 0)
				{
					verbose = (stoi(argv[12]) == 1) ? true : false;
				}
				TestGMPHDFilter tracker(_firstFrameFileName, _gtFileName, _detectionFile);
				tracker.run(verbose, dpp_epsilon);
			}
			else if(strcmp(argv[8], "nms") == 0){
				if(strcmp(argv[9], "-threshold") == 0){
					nms_threshold = stod(argv[10]);
				}
				else{
					cout << "No nms epsilon value given" << endl;
					cout << "exiting..." << endl;
					return EXIT_FAILURE;
				}
				if(strcmp(argv[11], "-neighbors") == 0){
					nms_neighbors = stoi(argv[12]);
				}
				else{
					cout << "No nms neighbors value given" << endl;
					cout << "exiting..." << endl;
					return EXIT_FAILURE;
				}
				if(strcmp(argv[13], "-minscores") == 0){
					nms_min_scores_num = stod(argv[14]);
				}
				else{
					cout << "No nms min scores sum value given" << endl;
					cout << "exiting..." << endl;
					return EXIT_FAILURE;
				}
				if(strcmp(argv[15], "-verbose") == 0)
				{
					verbose = (stoi(argv[16]) == 1) ? true : false;
				}
				TestGMPHDFilter tracker(_firstFrameFileName, _gtFileName, _detectionFile);
				tracker.run(verbose, nms_threshold, nms_neighbors, nms_min_scores_num);
			}
			else{
				cout << "Incorrect pruning method" << endl;
				cout << "exiting..." << endl;
				return EXIT_FAILURE;
			}
		}
		else{
			if(strcmp(argv[7], "-verbose") == 0)
			{
				verbose = (stoi(argv[8]) == 1) ? true : false;
			}
			TestGMPHDFilter tracker(_firstFrameFileName, _gtFileName, _detectionFile);
			tracker.run(verbose);
		}
	}
}