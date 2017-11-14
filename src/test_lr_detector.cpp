#include "test_lr_detector.hpp"

TestLRDetector::TestLRDetector(){}

TestLRDetector::TestLRDetector(string _firstFrameFileName, 
	string _groundTruthFileName, string _modelFilesPath,
	double _group_threshold, double _hit_threshold)
{
	this->firstFrameFileName = _firstFrameFileName;
	this->groundTruthFileName = _groundTruthFileName;
	this->modelFilesPath = _modelFilesPath;
	this->group_threshold = _group_threshold;
	this->hit_threshold = _hit_threshold;
	this->generator = ImageGenerator(this->firstFrameFileName, this->groundTruthFileName);
}

void TestLRDetector::run()
{
	//namedWindow("MTT", WINDOW_NORMAL);//WINDOW_NORMAL
	RNG rng( 0xFFFFFFFF );
	map<int,Scalar> color;
    CPU_LR_HOGDetector detector;

    C_utils utils;
	VectorXd mean;
	VectorXd std;
	VectorXd max;
	VectorXd min;
	VectorXd weights;
    VectorXd bias;
    utils.read_Labels(this->modelFilesPath + "means.csv", mean);
	utils.read_Labels(this->modelFilesPath + "weights.csv", weights);
	utils.read_Labels(this->modelFilesPath + "stds.csv", std);
	utils.read_Labels(this->modelFilesPath + "bias.csv", bias);
	utils.read_Labels(this->modelFilesPath + "maxs.csv", max);
	utils.read_Labels(this->modelFilesPath + "mins.csv", min);
    
    //detector.init(this->group_threshold, this->hit_threshold);
	detector.init(this->group_threshold, this->hit_threshold);
	detector.loadModel(weights, mean, std, max, min, bias(0));

	for (size_t i = 0; i < this->generator.getDatasetSize(); ++i)
	{
        Mat currentFrame = this->generator.getFrame(i);
		MatrixXd features;
		vector<double> weights;
		vector<Rect> preDetections;
		detector.detect(currentFrame, preDetections, weights, features);
		
		for(size_t j = 0; j < preDetections.size(); j++){
			cout << i + 1 << ",-1" << 
			preDetections.at(j).x << "," <<
			preDetections.at(j).y << "," <<
			preDetections.at(j).width << "," <<
			preDetections.at(j).height << "," <<
			weights[j] << ",-1,-1,-1";
			for(unsigned int k = 0; k < features.cols(); k++){
				cout << "," << features(j,k);
			}
			cout << endl;
		}
        /*for (size_t i = 0; i < preDetections.size(); ++i){
            rectangle( currentFrame, preDetections.at(i), Scalar(0,0,255), 2, LINE_8  );
        }
		imshow("MTT", currentFrame);
		waitKey(1);*/
	}
}

int main(int argc, char const *argv[])
{
	string _firstFrameFileName, _gtFileName, _modelFilesPath;
	double _group_threshold, _hit_threshold;
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
	    	_gtFileName = argv[4];
	  	}
	  	else
	  	{
	  		cout << "No ground truth given" << endl;
	  		cout << "exiting..." << endl;
	  		return EXIT_FAILURE;
	  	}
	  	if(strcmp(argv[5], "-model") == 0)
	  	{
	    	_modelFilesPath = argv[6];
	  	}
	  	else
	  	{
	  		cout << "No model folder path given" << endl;
	  		cout << "exiting..." << endl;
	  		return EXIT_FAILURE;
		}
		if(strcmp(argv[7], "-group") == 0)
		{
			_group_threshold = stod(argv[8]);
		}
		else
		{
			cout << "No group threshold given" << endl;
			cout << "exiting..." << endl;
			return EXIT_FAILURE;
		}
		if(strcmp(argv[9], "-hit") == 0)
		{
			_hit_threshold = stod(argv[10]);
		}
		else
		{
			cout << "No hit threshold given" << endl;
			cout << "exiting..." << endl;
			return EXIT_FAILURE;
		}
	  	TestLRDetector tracker(_firstFrameFileName, _gtFileName, _modelFilesPath, _group_threshold, _hit_threshold);
	  	tracker.run();
	}
}