#include "test_dpp_pets.hpp"

MultiTargetTrackingDPPPets::MultiTargetTrackingDPPPets(){}

MultiTargetTrackingDPPPets::MultiTargetTrackingDPPPets(string _firstFrameFileName, string _groundTruthFileName, int _npart)
{
	this->firstFrameFileName = _firstFrameFileName;
	this->groundTruthFileName = _groundTruthFileName;
	this->npart = _npart;

	this->dpp = DPP();
	this->generator = ImageGenerator(this->firstFrameFileName, this->groundTruthFileName);
	this->hogDetector = HOGDetector();
}

void MultiTargetTrackingDPPPets::run()
{
	namedWindow("MTT");
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
			filter.draw_particles(currentFrame, Scalar(255, 255, 255));
		}
		else
		{
			filter.predict();
			filter.update(currentFrame, detections);
			vector<Rect> estimates = filter.estimate(currentFrame, true);
			filter.draw_particles(currentFrame, Scalar(255, 255, 255));
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
	string _firstFrameFileName, _groundTruthFileName;
	int _npart;
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
	    	_groundTruthFileName = argv[4];
	  	}
	  	else
	  	{
	  		cout << "No groundtruth given" << endl;
	  		cout << "exiting..." << endl;
	  		return EXIT_FAILURE;
	  	}
	  	if (strcmp(argv[5], "-npart") == 0)
	  	{
	  		_npart = atoi(argv[6]);
	  	}
	  	else
	  	{
	  		cout << "No particles number given" << endl;
	  		cout << "exiting..." << endl;
	  		return EXIT_FAILURE;
	  	}
	  	MultiTargetTrackingDPPPets tracker(_firstFrameFileName, _groundTruthFileName, _npart);
	  	tracker.run();
	}
}