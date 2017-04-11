#include "test_phd_pets.hpp"

MultiTargetTrackingPHDFilterPets::MultiTargetTrackingPHDFilterPets(){}

MultiTargetTrackingPHDFilterPets::MultiTargetTrackingPHDFilterPets(string _firstFrameFileName, 
	string _groundTruthFileName, int _npart)
{
	this->firstFrameFileName = _firstFrameFileName;
	this->groundTruthFileName = _groundTruthFileName;
	this->npart = _npart;
	this->generator = ImageGenerator(this->firstFrameFileName, this->groundTruthFileName);
	this->hogDetector = HOGDetector();
}

void MultiTargetTrackingPHDFilterPets::run()
{
	namedWindow("MTT");
	RNG rng( 0xFFFFFFFF );
	map<int,Scalar> color;
	//resizeWindow("MTT", 400, 400);
	PHDParticleFilter filter(this->npart);

	Mat currentFrame;

	for (int i = 0; i < this->generator.getDatasetSize(); ++i)
	{	
		Mat currentFrame = this->generator.getFrame(i);
		vector<Target> gt = this->generator.getGroundTruth(i);

		vector<Rect> preDetections;
	
		preDetections = this->hogDetector.detect(currentFrame);
		
		cout << "--------------------------" << endl;
		cout << "groundtruth number: " << gt.size() << endl;
		cout << "preDetections number: " << preDetections.size() << endl;
		
		if (!filter.is_initialized())
		{
			filter.initialize(currentFrame, preDetections);
			filter.draw_particles(currentFrame, Scalar(255, 255, 255));
			vector<Target> estimates = filter.estimate(currentFrame, true);
		}
		else
		{
			filter.predict();
			filter.update(currentFrame, preDetections);
			//filter.draw_particles(currentFrame, Scalar(255, 255, 255));
			vector<Target> estimates = filter.estimate(currentFrame, true);
		}

		for (size_t j = 0; j < preDetections.size(); ++j)
		{
			rectangle(currentFrame, preDetections.at(j), Scalar(0,255,0), 2, LINE_AA);
		}
		
		
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
	  	MultiTargetTrackingPHDFilterPets tracker(_firstFrameFileName, _groundTruthFileName, _npart);
	  	tracker.run();
	}
}