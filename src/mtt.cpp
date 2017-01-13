#include "mtt.hpp"

#ifndef PARAMS
const bool CNN_FEATURES = true;	
#endif

MultiTargetTracking::MultiTargetTracking(){}

MultiTargetTracking::MultiTargetTracking(string _firstFrameFileName, string _groundTruthFileName, int _npart)
{
	this->firstFrameFileName = _firstFrameFileName;
	this->groundTruthFileName = _groundTruthFileName;
	this->npart = _npart;

	this->dpp = DPP();
	this->generator = ImageGenerator(this->firstFrameFileName, this->groundTruthFileName);
	this->hogDetector = HOGDetector();

	initialized = true;
}

MultiTargetTracking::MultiTargetTracking(string _firstFrameFileName, string _groundTruthFileName, string _firstCNNFeaturesFile, string _firstPreDetectionFile, int _npart)
{
	this->firstFrameFileName = _firstFrameFileName;
	this->groundTruthFileName = _groundTruthFileName;
	this->firstCNNFeaturesFile = _firstCNNFeaturesFile;
	this->firstPreDetectionFile = _firstPreDetectionFile;
	this->npart = _npart;

	this->dpp = DPP();
	this->generator = ImageGenerator(this->firstFrameFileName, this->groundTruthFileName);
	this->cnnReader = CNNReader(this->firstCNNFeaturesFile, this->firstPreDetectionFile);
	
	initialized = true;
}

void MultiTargetTracking::run()
{
	namedWindow("MTT");
	particle_filter filter(this->npart);

	for (unsigned int i = 0; i < generator.getDatasetSize(); ++i)
	{
		Mat currentFrame = generator.getFrame(i);
		vector<Target> gt = generator.getGroundTruth(i);

		MatrixXd features; vector<Rect> preDetections; VectorXd detectionWeights;

		if(CNN_FEATURES){
			features = this->cnnReader.getFeatureValues();
			preDetections = this->cnnReader.getPreDetections();
			detectionWeights = this->cnnReader.getDetectionWeights();
		}
		else{
			preDetections = this->hogDetector.detect(currentFrame);
			features = this->hogDetector.getFeatureValues();
			detectionWeights = this->hogDetector.getDetectionWeights();
		}

		/*cout << "features size: " << features.rows() << "," << features.cols() << endl;
		cout << "preDetections size: " << preDetections.size() << endl;
		cout << "detectionWeights size: " << detectionWeights.size() << endl;*/

		double alpha = 0.9, beta = 1.1, lambda = -0.1, mu = 0.8, epsilon = 0.1;
		vector<Rect> detections = this->dpp.run(preDetections, detectionWeights, features, alpha, lambda, beta, mu, epsilon);

		cout << "--------------------------" << endl;
		cout << "groundtruth number: " << gt.size() << endl;
		cout << "detections number: " << detections.size() << endl;
		
		if (!filter.is_initialized())
		{
			filter.initialize(currentFrame, detections);
		}
		else
		{
			filter.predict();
			filter.update(currentFrame, detections);
			vector<Rect> estimates = filter.estimate(currentFrame, true);
			filter.draw_particles(currentFrame, Scalar(0, 255, 255));
			cout << "estimate number: " << estimates.size() << endl;
		}

		for (size_t j = 0; j < detections.size(); ++j)
		{
			rectangle(currentFrame, detections.at(j), Scalar(0,255,0), 1, LINE_AA);
		}
		/*for (unsigned int j = 0; j < gt.size(); ++j)
		{
			rectangle(currentFrame, gt.at(j).bbox, Scalar(0,255,0), 1, LINE_AA);
		}*/		
		
		imshow("MTT", currentFrame);
		waitKey(1);
		
		
	}
}

int main(int argc, char const *argv[])
{
	string _firstFrameFileName, _gtFileName, _firstCNNFeaturesFile, _firstPreDetectionFile;
	int _npart;
	if(CNN_FEATURES){
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

		  	if(strcmp(argv[5], "-cnn") == 0)
		  	{
		    	_firstCNNFeaturesFile = argv[6];
		  	}
		  	else
		  	{
		  		cout << "No CNN path given" << endl;
		  		cout << "exiting..." << endl;
		  		return EXIT_FAILURE;
		  	}
		  	if(strcmp(argv[7], "-pd") == 0)
		  	{
		    	_firstPreDetectionFile = argv[8];
		  	}
		  	else
		  	{
		  		cout << "No preDetections path given" << endl;
		  		cout << "exiting..." << endl;
		  		return EXIT_FAILURE;
		  	}
		  	if (strcmp(argv[9], "-npart") == 0)
		  	{
		  		_npart = atoi(argv[10]);
		  	}
		  	else
		  	{
		  		cout << "No number particles given" << endl;
		  		cout << "exiting..." << endl;
		  		return EXIT_FAILURE;
		  	}
		  	MultiTargetTracking tracker(_firstFrameFileName, _gtFileName, _firstCNNFeaturesFile, _firstPreDetectionFile, _npart);
		  	tracker.run();
		}
	}
	else{
		if(argc != 7)
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
		  	if (strcmp(argv[5], "-npart") == 0)
		  	{
		  		_npart = atoi(argv[6]);
		  	}
		  	else
		  	{
		  		cout << "No number particles given" << endl;
		  		cout << "exiting..." << endl;
		  		return EXIT_FAILURE;
		  	}
		  	MultiTargetTracking tracker(_firstFrameFileName, _gtFileName, _npart);
		  	tracker.run();
		}
	}
	
}