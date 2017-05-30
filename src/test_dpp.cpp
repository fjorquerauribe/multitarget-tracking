#include "test_dpp.hpp"

MultiTargetTrackingDPP::MultiTargetTrackingDPP(){}

MultiTargetTrackingDPP::MultiTargetTrackingDPP(string _firstFrameFileName, string _groundTruthFileName, string _preDetectionFile,
	double _epsilon, double _mu, double _lambda, int _npart)
{
	this->firstFrameFileName = _firstFrameFileName;
	this->groundTruthFileName = _groundTruthFileName;
	this->preDetectionFile = _preDetectionFile;
	this->npart = _npart;
	this->epsilon = _epsilon;
	this->mu = _mu;
	this->lambda = _lambda;
	this->generator = ImageGenerator(this->firstFrameFileName, this->groundTruthFileName, this->preDetectionFile);
}

void MultiTargetTrackingDPP::run()
{
	namedWindow("MTT", WINDOW_NORMAL);

#ifdef WITH_CUDA
	CUDA_HOGDetector hogDetector = CUDA_HOGDetector(0, 0.0);
	hogDetector.loadPreTrainedModel();
#else
	HOGDetector hogDetector = HOGDetector(0, 0.0);
#endif

	resizeWindow("MTT", 400, 400);
	PHDParticleFilter filter(this->npart);
	DPP dpp = DPP();

	for (size_t i = 0; i < this->generator.getDatasetSize(); ++i)
	{
		Mat currentFrame = this->generator.getFrame(i);
		vector<Target> gt = this->generator.getGroundTruth(i);

		MatrixXd features; vector<Rect> preDetections; VectorXd detectionWeights;
	
		preDetections = hogDetector.detect(currentFrame);
		features = hogDetector.getFeatureValues();
		detectionWeights = hogDetector.getDetectionWeights();
		//preDetections = this->generator.getDetections(i);

		vector<Rect> detections = dpp.run(preDetections, detectionWeights, features, this->epsilon, this->mu, this->lambda);

		/*cout << "--------------------------" << endl;
		cout << "groundtruth number: " << gt.size() << endl;
		cout << "detections number: " << detections.size() << endl;*/
		vector<Target> estimates;
		if (!filter.is_initialized())
		{
			filter.initialize(currentFrame, detections);
		}
		else
		{
			filter.predict();
			filter.update(currentFrame, detections);
			estimates = filter.estimate(currentFrame, true);
			//filter.draw_particles(currentFrame, Scalar(0, 255, 255));
			//cout << "estimate number: " << estimates.size() << endl;
		}

		/*for (size_t j = 0; j < detections.size(); ++j)
		{
			rectangle(currentFrame, detections.at(j), Scalar(0,255,0), 1, LINE_AA);
		}
		for (unsigned int j = 0; j < gt.size(); ++j)
		{
			rectangle(currentFrame, gt.at(j).bbox, Scalar(0,255,0), 1, LINE_AA);
		}*/
		for (size_t j = 0; j < estimates.size(); ++j)
		{
			cout << i << "," << estimates.at(j).bbox.x << "," << estimates.at(j).bbox.y << "," << 
			estimates.at(j).bbox.width << "," << estimates.at(j).bbox.height << endl;
		}
		
		imshow("MTT", currentFrame);
		waitKey(1);
		
		
	}
}

int main(int argc, char const *argv[])
{
	string _firstFrameFileName, _gtFileName, _preDetectionFile;
	int _npart;
	double _epsilon, _mu, _lambda;
	if(argc != 15)
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
	    	_preDetectionFile = argv[6];
	  	}
	  	else
	  	{
	  		cout << "No detections file given" << endl;
	  		cout << "exiting..." << endl;
	  		return EXIT_FAILURE;
	  	}
	  	if(strcmp(argv[7], "-epsilon") == 0)
	  	{
	    	_epsilon = stod(argv[8]);
	  	}
	  	else
	  	{
	  		cout << "No epsilon given" << endl;
	  		cout << "exiting..." << endl;
	  		return EXIT_FAILURE;
	  	}
	  	if(strcmp(argv[9], "-mu") == 0)
	  	{
	    	_mu = stod(argv[10]);
	  	}
	  	else
	  	{
	  		cout << "No mu given" << endl;
	  		cout << "exiting..." << endl;
	  		return EXIT_FAILURE;
	  	}
	  	if(strcmp(argv[11], "-lambda") == 0)
	  	{
	    	_lambda = stod(argv[12]);
	  	}
	  	else
	  	{
	  		cout << "No lambda given" << endl;
	  		cout << "exiting..." << endl;
	  		return EXIT_FAILURE;
	  	}
	  	if (strcmp(argv[13], "-npart") == 0)
	  	{
	  		_npart = atoi(argv[14]);
	  	}
	  	else
	  	{
	  		cout << "No particles number given" << endl;
	  		cout << "exiting..." << endl;
	  		return EXIT_FAILURE;
	  	}
	  	MultiTargetTrackingDPP tracker(_firstFrameFileName, _gtFileName, _preDetectionFile, _epsilon, _mu, _lambda, _npart);
	  	tracker.run();
	}
}