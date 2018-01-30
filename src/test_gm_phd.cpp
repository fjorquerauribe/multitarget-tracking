#include "test_gm_phd.hpp"

TestGMPHDFilter::TestGMPHDFilter(){}

TestGMPHDFilter::TestGMPHDFilter(string _firstFrameFileName, 
	string _groundTruthFileName, string _detectionFile)
{
	this->firstFrameFileName = _firstFrameFileName;
	this->groundTruthFileName = _groundTruthFileName;
	this->detectionFile = _detectionFile;
	this->generator = ImageGenerator(this->firstFrameFileName, this->groundTruthFileName, this->detectionFile);
}

void TestGMPHDFilter::run(bool verbose = false)
{
	RNG rng( 0xFFFFFFFF );
	map<int,Scalar> color;
	PHDGaussianMixture filter(verbose);
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
			imwrite("gm_phd_" + to_string(i) + ".png", currentFrame);
			waitKey(100);
		}
	}
}

int main(int argc, char const *argv[])
{
	string _firstFrameFileName, _gtFileName, _detectionFile;
	bool verbose = false;
	if(argc != 9)
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
		if(strcmp(argv[7], "-verbose") == 0)
	  	{
	    	verbose = (stoi(argv[8]) == 1) ? true : false;
	  	}
	  	TestGMPHDFilter tracker(_firstFrameFileName, _gtFileName, _detectionFile);
	  	tracker.run(verbose);
	}
	/*
	string _firstFrameFileName, _gtFileName, _detectionFile;
	bool verbose = false;
	if(argc != 11)
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
		if(strcmp(argv[7], "-verbose") == 0)
	  	{
	    	verbose = (stoi(argv[8]) == 1) ? true : false;
	  	}
	  	TestGMPHDFilter tracker(_firstFrameFileName, _gtFileName, _detectionFile);
	  	tracker.run(verbose);
	}
	*/
}