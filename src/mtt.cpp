#include "mtt.hpp"

MultiTargetTracking::MultiTargetTracking(){}

MultiTargetTracking::MultiTargetTracking(string _firstFrameFileName, string _groundTruthFileName)
{
	this->firstFrameFileName = _firstFrameFileName;
	this->groundTruthFileName = _groundTruthFileName;
	this->hogDetector = HOGDetector();
	this->dpp = DPP();
	initialized = true;
}

void MultiTargetTracking::run()
{
	ImageGenerator generator(this->firstFrameFileName, this->groundTruthFileName);

	namedWindow("MTT");
	for (unsigned int i = 0; i < generator.getDatasetSize(); ++i)
	{
		Mat currentFrame = generator.getFrame(i);
		vector<Target> gt = generator.getGroundTruth(i);

		vector<Rect> hogDetections = this->hogDetector.detect(currentFrame);
		MatrixXd hogFeatures = this->hogDetector.getFeatureValues();
		VectorXd detectionWeights = this->hogDetector.getWeightDetections();

		double alpha = 0.9, lambda = 1.1, beta = 0.1, mu = 0.8, epsilon = 0.1;
		this->dpp.run(hogDetections, detectionWeights, hogFeatures, alpha, lambda, beta, mu, epsilon);

		for (unsigned int j = 0; j < gt.size(); ++j)
		{
			rectangle(currentFrame, gt.at(j).bbox, Scalar(0,255,0), 1, LINE_AA);
		}
		this->hogDetector.draw();
		//cout << "groundtruth size: " << gt.size() << "\t";
		
		imshow("MTT", currentFrame);
		waitKey(1);
	}
}

int main(int argc, char const *argv[])
{
	string _firstFrameFileName,_gtFileName;
	if(argc != 5)
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
	  	MultiTargetTracking tracker(_firstFrameFileName, _gtFileName);
	  	tracker.run();
	}
}