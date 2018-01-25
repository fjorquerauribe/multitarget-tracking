#include "test_dpp.hpp"

TestDPP::TestDPP(){}

TestDPP::TestDPP(string _firstFrameFileName, string _groundTruthFileName, string _preDetectionFile,
	double _epsilon, double _mu, double _lambda)
{
	this->firstFrameFileName = _firstFrameFileName;
	this->groundTruthFileName = _groundTruthFileName;
	this->preDetectionFile = _preDetectionFile;
	this->epsilon = _epsilon;
	this->mu = _mu;
	this->lambda = _lambda;
	this->generator = ImageGenerator(this->firstFrameFileName, this->groundTruthFileName, this->preDetectionFile);
}

void TestDPP::run(bool verbose = false)
{
	if(verbose) namedWindow("MTT", WINDOW_NORMAL);

	DPP dpp = DPP();

	for (size_t i = 0; i < this->generator.getDatasetSize(); ++i)
	{
		Mat currentFrame = this->generator.getFrame(i);
		vector<Target> gt = this->generator.getGroundTruth(i);

		vector<Rect> preDetections;
		VectorXd detectionWeights;
		MatrixXd features;

		preDetections = this->generator.getDetections(i);
		detectionWeights = this->generator.getDetectionWeights(i);
		features = this->generator.getDetectionFeatures(i);

		//vector<Rect> detections = dpp.run(preDetections, detectionWeights, features, this->epsilon, this->mu, this->lambda);
		
		vector<Rect> detections;
		vector<double> scores(detectionWeights.data(), detectionWeights.data() + detectionWeights.size());
		nms2(preDetections, scores, detections, 0.8);

		for(size_t j = 0; j < detections.size(); j++){
			cout << i + 1 
			<< ",-1"
			<< "," << detections.at(j).x
			<< "," << detections.at(j).y
			<< "," << detections.at(j).width
			<< "," << detections.at(j).height
			<< ",1,-1,-1,-1" << endl;
			rectangle( currentFrame, detections.at(j), Scalar(255, 0, 0), 3, LINE_8  );
		}
		
		if(verbose) {
			cout << "Target number: " << gt.size() << endl;
			cout << "preDetections number: " << preDetections.size() << endl;
			cout << "Detections number: " << detections.size() << endl;
			cout << "----------------------------------------" << endl;

			imshow("MTT", currentFrame);
			waitKey(1);
		}

	}
}

int main(int argc, char const *argv[])
{
	string _firstFrameFileName, _gtFileName, _preDetectionFile;
	double _epsilon, _mu, _lambda;
	bool verbose;
	
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
	  	if (strcmp(argv[13], "-verbose") == 0)
	  	{
	  		verbose = (stoi(argv[14]) == 1) ? true : false;
		}
	  	TestDPP tracker(_firstFrameFileName, _gtFileName, _preDetectionFile, _epsilon, _mu, _lambda);
	  	tracker.run(verbose);
	}
}