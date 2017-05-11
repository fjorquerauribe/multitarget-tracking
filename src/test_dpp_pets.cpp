#include "test_dpp_pets.hpp"

MultiTargetTrackingDPPPets::MultiTargetTrackingDPPPets(){}

MultiTargetTrackingDPPPets::MultiTargetTrackingDPPPets(string _firstFrameFileName, string _groundTruthFileName,
	double _epsilon, double _mu, double _lambda, int _npart)
{
	this->firstFrameFileName = _firstFrameFileName;
	this->groundTruthFileName = _groundTruthFileName;
	this->npart = _npart;
	this->epsilon = _epsilon;
	this->mu = _mu;
	this->lambda = _lambda;
	this->generator = ImageGenerator(this->firstFrameFileName, this->groundTruthFileName);
}

void MultiTargetTrackingDPPPets::run()
{
	namedWindow("MTT");
	PHDParticleFilter filter(this->npart);
	HOGDetector hogDetector = HOGDetector(0, 0.0);
	DPP dpp = DPP();

	for (unsigned int i = 0; i < this->generator.getDatasetSize(); ++i)
	{
		Mat currentFrame = this->generator.getFrame(i);
		vector<Target> gt = this->generator.getGroundTruth(i);

		MatrixXd features; vector<Rect> preDetections; VectorXd detectionWeights;
	
		preDetections = hogDetector.detect(currentFrame);
		features = hogDetector.getFeatureValues();
		detectionWeights = hogDetector.getDetectionWeights();
		
		
		vector<Rect> detections = dpp.run(preDetections, detectionWeights, features, this->epsilon, this->mu, this->lambda);

		/*cout << "--------------------------" << endl;
		cout << "groundtruth number: " << gt.size() << endl;
		cout << "preDetections number: " << preDetections.size() << endl;
		cout << "detections number: " << detections.size() << endl;*/
		vector<Target> estimates;
		if (!filter.is_initialized())
		{
			filter.initialize(currentFrame, detections);
			filter.draw_particles(currentFrame, Scalar(255, 255, 255));
			estimates = filter.estimate(currentFrame, true);
		}
		else
		{
			filter.predict();
			filter.update(currentFrame, detections);
			filter.draw_particles(currentFrame, Scalar(255, 255, 255));
			estimates = filter.estimate(currentFrame, true);
			//cout << "estimate number: " << estimates.size() << endl;
		}
		
		/*for (unsigned int j = 0; j < gt.size(); ++j)
		{
			rectangle(currentFrame, gt.at(j).bbox, Scalar(0,255,0), 1, LINE_AA);
		}*/
		for (int j = 0; j < estimates.size(); ++j)
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
	string _firstFrameFileName, _groundTruthFileName;
	int _npart;
	double _epsilon, _mu, _lambda;
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
	    	_groundTruthFileName = argv[4];
	  	}
	  	else
	  	{
	  		cout << "No groundtruth given" << endl;
	  		cout << "exiting..." << endl;
	  		return EXIT_FAILURE;
	  	}
	  	if(strcmp(argv[5], "-epsilon") == 0)
	  	{
	    	_epsilon = stod(argv[6]);
	  	}
	  	else
	  	{
	  		cout << "No epsilon given" << endl;
	  		cout << "exiting..." << endl;
	  		return EXIT_FAILURE;
	  	}
	  	if(strcmp(argv[7], "-mu") == 0)
	  	{
	    	_mu = stod(argv[8]);
	  	}
	  	else
	  	{
	  		cout << "No mu given" << endl;
	  		cout << "exiting..." << endl;
	  		return EXIT_FAILURE;
	  	}
	  	if(strcmp(argv[9], "-lambda") == 0)
	  	{
	    	_lambda = stod(argv[10]);
	  	}
	  	else
	  	{
	  		cout << "No lambda given" << endl;
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
	  	MultiTargetTrackingDPPPets tracker(_firstFrameFileName, _groundTruthFileName, _epsilon, _mu, _lambda, _npart);
	  	tracker.run();
	}
}