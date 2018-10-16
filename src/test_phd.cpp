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

TestPHDFilter::TestPHDFilter(string _firstFrameFileName, string _groundTruthFileName, string model_cfg,
	string model_binary, string class_names, float min_confidence, int _npart)
{
	this->firstFrameFileName = _firstFrameFileName;
	this->groundTruthFileName = _groundTruthFileName;
	this->model_cfg = model_cfg;
	this->model_binary = model_binary;
	this->class_names = class_names;
	this->min_confidence = min_confidence;
	this->npart = _npart;
	this->detector = YOLODetector(this->model_cfg, this->model_binary, this->class_names, this->min_confidence);
	this->generator = ImageGenerator(this->firstFrameFileName, this->groundTruthFileName);
}

void TestPHDFilter::run()
{
	RNG rng( 0xFFFFFFFF );
	map<int,Scalar> color;
	bool verbose = false;
	PHDParticleFilter filter(this->npart, verbose);
	if(verbose) namedWindow("MTT", WINDOW_NORMAL);//WINDOW_NORMAL
	vector<Rect> detections;
	
	for (size_t i = 0; i < this->generator.getDatasetSize(); ++i)
	{
		Mat currentFrame = this->generator.getFrame(i);
		vector<MyTarget> gt = this->generator.getGroundTruth(i);
		
		if(this->detector.is_initialized()) 
		{	
			detections = this->detector.detect(currentFrame);
		}	
		else
		{
			detections = this->generator.getDetections(i);
		}

		vector<MyTarget> estimates;
		
		if(verbose)	cout << "Target number: " << gt.size() << endl;

		if (!filter.is_initialized() &&  gt.size()>0)
		{
			filter.initialize(currentFrame, detections);
			estimates = filter.estimate(currentFrame, true);
			//filter.draw_particles(currentFrame, Scalar(255, 255, 255));
		}
		else
		{
			filter.predict();
			filter.update(currentFrame, detections);
			estimates = filter.estimate(currentFrame, true);
			//filter.draw_particles(currentFrame, Scalar(255, 255, 255));
		}

		for(size_t j = 0; j < estimates.size(); j++){
			cout << i + 1
			<< "," << estimates.at(j).label
			<< "," << estimates.at(j).bbox.x
			<< "," << estimates.at(j).bbox.y
			<< "," << estimates.at(j).bbox.width
			<< "," << estimates.at(j).bbox.height
			<< ",1,-1,-1,-1" << endl;
		}
		detections.clear();
		//cout << "----------------------------------------" << endl;
		imshow("MTT", currentFrame);
		waitKey(1);
	}
}

int main(int argc, char const *argv[])
{
	string _firstFrameFileName, _gtFileName, _preDetectionFile, 
		model_cfg, model_binary, class_names;
	float min_confidence;
	int _npart;
	if(argc != 9 && argc != 15)
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
	  	else
	  	{
			if(strcmp(argv[5], "-config") == 0){
				model_cfg = argv[6];
			}
			else
			{
				cout << "No model configuration given" << endl;
				cout << "exiting..." << endl;
				return EXIT_FAILURE;
			}
			if(strcmp(argv[7], "-model") == 0){
				model_binary = argv[8];
			}
			else
			{
				cout << "No model given" << endl;
				cout << "exiting..." << endl;
				return EXIT_FAILURE;
			}
			if(strcmp(argv[9], "-classes") == 0){
				class_names = argv[10];
			}
			else
			{
				cout << "No class names given" << endl;
				cout << "exiting..." << endl;
				return EXIT_FAILURE;
			}
			if(strcmp(argv[11], "-min_confidence") == 0){
				min_confidence = stod(argv[12]);
			}
			else
			{
				cout << "No min confidence given" << endl;
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
			TestPHDFilter tracker(_firstFrameFileName, _gtFileName, model_cfg, model_binary, class_names, min_confidence, _npart);
			tracker.run();
	  	}  	
	}
}