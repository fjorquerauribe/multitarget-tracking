#include "test_phd.hpp"

TestPHDFilter::TestPHDFilter(){}

TestPHDFilter::TestPHDFilter(string _firstFrameFileName, 
	string _groundTruthFileName, string _preDetectionFile, int _npart)
{
	this->firstFrameFileName = _firstFrameFileName;
	this->groundTruthFileName = _groundTruthFileName;
	this->preDetectionFile = _preDetectionFile;
	this->npart = _npart;
	this->generator = ImageGenerator(this->firstFrameFileName, this->groundTruthFileName, this->preDetectionFile);
}

void TestPHDFilter::run()
{
	namedWindow("MTT", WINDOW_NORMAL);//WINDOW_NORMAL
	RNG rng( 0xFFFFFFFF );
	map<int,Scalar> color;
	PHDParticleFilter filter(this->npart);

	for (size_t i = 0; i < this->generator.getDatasetSize(); ++i)
	{
		Mat currentFrame = this->generator.getFrame(i);
		vector<Target> gt = this->generator.getGroundTruth(i);

		vector<Rect> preDetections = this->generator.getDetections(i);

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
			estimates = filter.estimate(currentFrame, true);
		}
		
		imshow("MTT", currentFrame);
		waitKey(1);
	}
}

int main(int argc, char const *argv[])
{
	string _firstFrameFileName, _gtFileName, _preDetectionFile;
	int _npart;
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
	    	_preDetectionFile = argv[6];
	  	}
	  	else
	  	{
	  		cout << "No detections file given" << endl;
	  		cout << "exiting..." << endl;
	  		return EXIT_FAILURE;
	  	}
	  	if (strcmp(argv[7], "-npart") == 0)
	  	{
	  		_npart = atoi(argv[8]);
	  	}
	  	else
	  	{
	  		cout << "No particles number given" << endl;
	  		cout << "exiting..." << endl;
	  		return EXIT_FAILURE;
	  	}
	  	TestPHDFilter tracker(_firstFrameFileName, _gtFileName, _preDetectionFile, _npart);
	  	tracker.run();
	}
}