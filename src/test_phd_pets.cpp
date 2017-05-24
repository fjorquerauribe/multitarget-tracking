#include "test_phd_pets.hpp"

MultiTargetTrackingPHDFilterPets::MultiTargetTrackingPHDFilterPets(){}

MultiTargetTrackingPHDFilterPets::MultiTargetTrackingPHDFilterPets(string _firstFrameFileName, 
	string _groundTruthFileName, int _group_threshold, double _hit_threshold, int _npart)
{
	this->firstFrameFileName = _firstFrameFileName;
	this->groundTruthFileName = _groundTruthFileName;
	this->npart = _npart;
	this->group_threshold = _group_threshold;
	this->hit_threshold = _hit_threshold;
	this->generator = ImageGenerator(this->firstFrameFileName, this->groundTruthFileName);
}

void MultiTargetTrackingPHDFilterPets::run()
{
	namedWindow("MTT");
	RNG rng( 0xFFFFFFFF );
	map<int,Scalar> color;
	//resizeWindow("MTT", 400, 400);
	PHDParticleFilter filter(this->npart);

#ifdef WITH_CUDA
	CUDA_HOGDetector hogDetector = CUDA_HOGDetector(this->group_threshold, this->hit_threshold;
#else
	HOGDetector hogDetector = HOGDetector(this->group_threshold, this->hit_threshold);
#endif

	Mat currentFrame;

	for (int i = 0; i < this->generator.getDatasetSize(); ++i)
	{	
		Mat currentFrame = this->generator.getFrame(i);
		vector<Target> gt = this->generator.getGroundTruth(i);

		vector<Rect> preDetections = hogDetector.detect(currentFrame);
		/*cout << "--------------------------" << endl;
		cout << "groundtruth number: " << gt.size() << endl;
		cout << "preDetections number: " << preDetections.size() << endl;*/
		vector<Target> estimates;
		if (!filter.is_initialized())
		{
			filter.initialize(currentFrame, preDetections);
			filter.draw_particles(currentFrame, Scalar(255, 255, 255));
			estimates = filter.estimate(currentFrame, true);
		}
		else
		{
			filter.predict();
			filter.update(currentFrame, preDetections);
			//filter.draw_particles(currentFrame, Scalar(255, 255, 255));
			estimates = filter.estimate(currentFrame, true);
		}

		for (size_t j = 0; j < preDetections.size(); ++j)
		{
			rectangle(currentFrame, preDetections.at(j), Scalar(0,255,0), 2, LINE_AA);
		}
		
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
	double _group_threshold, _hit_threshold;
	int _npart;
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
	    	_groundTruthFileName = argv[4];
	  	}
	  	else
	  	{
	  		cout << "No groundtruth given" << endl;
	  		cout << "exiting..." << endl;
	  		return EXIT_FAILURE;
	  	}
	  	if(strcmp(argv[5], "-gp_t") == 0)
	  	{
	    	_group_threshold = stod(argv[6]);
	  	}
	  	else
	  	{
	  		cout << "No group threshold given" << endl;
	  		cout << "exiting..." << endl;
	  		return EXIT_FAILURE;
	  	}
	  	if(strcmp(argv[7], "-hit_t") == 0)
	  	{
	    	_hit_threshold = stod(argv[8]);
	  	}
	  	else
	  	{
	  		cout << "No hit threshold given" << endl;
	  		cout << "exiting..." << endl;
	  		return EXIT_FAILURE;
	  	}
	  	if (strcmp(argv[9], "-npart") == 0)
	  	{
	  		_npart = atoi(argv[10]);
	  	}
	  	else
	  	{
	  		cout << "No particles number given" << endl;
	  		cout << "exiting..." << endl;
	  		return EXIT_FAILURE;
	  	}
	  	MultiTargetTrackingPHDFilterPets tracker(_firstFrameFileName, _groundTruthFileName, _group_threshold, _hit_threshold, _npart);
	  	tracker.run();
	}
}