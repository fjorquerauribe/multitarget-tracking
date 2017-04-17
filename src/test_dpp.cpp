#include "test_dpp.hpp"

MultiTargetTrackingDPP::MultiTargetTrackingDPP(){}

MultiTargetTrackingDPP::MultiTargetTrackingDPP(string _firstFrameFileName, string _groundTruthFileName, string _preDetectionFile,
	double group_threshold, double hit_threshold, int _npart)
{
	this->firstFrameFileName = _firstFrameFileName;
	this->groundTruthFileName = _groundTruthFileName;
	this->preDetectionFile = _preDetectionFile;
	this->npart = _npart;

	this->dpp = DPP();
	this->generator = ImageGenerator(this->firstFrameFileName, this->groundTruthFileName, this->preDetectionFile);
	this->hogDetector = HOGDetector(group_threshold, hit_threshold);
}

void MultiTargetTrackingDPP::run()
{
	namedWindow("MTT", WINDOW_NORMAL);
	//resizeWindow("MTT", 400, 400);
	PHDParticleFilter filter(this->npart);

	for (unsigned int i = 0; i < this->generator.getDatasetSize(); ++i)
	{
		Mat currentFrame = this->generator.getFrame(i);
		vector<Target> gt = this->generator.getGroundTruth(i);

		MatrixXd features; vector<Rect> preDetections; VectorXd detectionWeights;
	
		preDetections = this->hogDetector.detect(currentFrame);
		features = this->hogDetector.getFeatureValues();
		detectionWeights = this->hogDetector.getDetectionWeights();
		//preDetections = this->generator.getDetections(i);

		/*cout << "features size: " << features.rows() << "," << features.cols() << endl;
		cout << "preDetections size: " << preDetections.size() << endl;
		cout << "detectionWeights size: " << detectionWeights.size() << endl;*/

		vector<Rect> detections = this->dpp.run(preDetections, detectionWeights, features);

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
			vector<Target> estimates = filter.estimate(currentFrame, true);
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
	string _firstFrameFileName, _gtFileName, _preDetectionFile;
	double _group_threshold, _hit_threshold;
	int _npart;
	if(argc != 13)
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
	  	if(strcmp(argv[7], "-gp_t") == 0)
	  	{
	    	_group_threshold = stod(argv[8]);
	  	}
	  	else
	  	{
	  		cout << "No group threshold given" << endl;
	  		cout << "exiting..." << endl;
	  		return EXIT_FAILURE;
	  	}
	  	if(strcmp(argv[9], "-hit_t") == 0)
	  	{
	    	_hit_threshold = stod(argv[10]);
	  	}
	  	else
	  	{
	  		cout << "No hit threshold given" << endl;
	  		cout << "exiting..." << endl;
	  		return EXIT_FAILURE;
	  	}
	  	if (strcmp(argv[11], "-npart") == 0)
	  	{
	  		_npart = atoi(argv[12]);
	  	}
	  	else
	  	{
	  		cout << "No particles number given" << endl;
	  		cout << "exiting..." << endl;
	  		return EXIT_FAILURE;
	  	}
	  	MultiTargetTrackingDPP tracker(_firstFrameFileName, _gtFileName, _preDetectionFile, _group_threshold, _hit_threshold, _npart);
	  	tracker.run();
	}
}