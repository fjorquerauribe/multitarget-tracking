#include "test_phd.hpp"

MultiTargetTrackingPHDFilter::MultiTargetTrackingPHDFilter(){}

MultiTargetTrackingPHDFilter::MultiTargetTrackingPHDFilter(string _firstFrameFileName, 
	string _groundTruthFileName, string _preDetectionFile, int _npart)
{
	this->firstFrameFileName = _firstFrameFileName;
	this->groundTruthFileName = _groundTruthFileName;
	this->preDetectionFile = _preDetectionFile;
	this->npart = _npart;
	this->generator = ImageGenerator(this->firstFrameFileName, this->groundTruthFileName, this->preDetectionFile);
}

void MultiTargetTrackingPHDFilter::run()
{
	namedWindow("MTT");
	RNG rng( 0xFFFFFFFF );
	map<int,Scalar> color;
	//resizeWindow("MTT", 400, 400);
	PHDParticleFilter filter(this->npart);

	for (int i = 0; i < this->generator.getDatasetSize(); ++i)
	{
		Mat currentFrame = this->generator.getFrame(i);
		vector<Target> gt = this->generator.getGroundTruth(i);

		MatrixXd features; vector<Rect> preDetections; VectorXd detectionWeights;
		preDetections = this->generator.getDetections(i);
		/*cout << "features size: " << features.rows() << "," << features.cols() << endl;
		cout << "preDetections size: " << preDetections.size() << endl;
		cout << "detectionWeights size: " << detectionWeights.size() << endl;*/
		
		cout << "--------------------------" << endl;
		cout << "groundtruth number: " << gt.size() << endl;
		cout << "preDetections number: " << preDetections.size() << endl;

		if (!filter.is_initialized())
		{
			
			filter.initialize(currentFrame, preDetections);
			filter.draw_particles(currentFrame, Scalar(255, 255, 255));
			vector<Rect> estimates = filter.estimate(currentFrame, true);
		}
		else
		{
			filter.predict();
			filter.update(currentFrame, preDetections);
			//filter.draw_particles(currentFrame, Scalar(255, 255, 255));
			vector<Rect> estimates = filter.estimate(currentFrame, true);
			cout << "estimate number: " << estimates.size() << endl;
		}

		for (size_t j = 0; j < preDetections.size(); ++j)
		{
			rectangle(currentFrame, preDetections.at(j), Scalar(0,255,0), 2, LINE_AA);
		}
		/*for (unsigned int j = 0; j < gt.size(); ++j)
		{
			if ( color.find(gt.at(j).label) == color.end() ) {
              int icolor = (unsigned) rng;
              Scalar new_color=Scalar( icolor&255, (icolor>>8)&255, (icolor>>16)&255 );
              color.insert (pair<int,Scalar>(gt.at(j).label,  new_color));
            }
            rectangle( currentFrame, gt.at(j).bbox, color.at(gt.at(j).label), 2, 1 );
		}*/
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
	  	MultiTargetTrackingPHDFilter tracker(_firstFrameFileName, _gtFileName, _preDetectionFile, _npart);
	  	tracker.run();
	}
}