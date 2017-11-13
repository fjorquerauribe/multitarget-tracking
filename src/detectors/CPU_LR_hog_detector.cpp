#include "CPU_LR_hog_detector.hpp"

#ifndef PARAMS
const bool USE_COLOR=false;
#endif

void CPU_LR_HOGDetector::init(double group_threshold, double hit_threshold){
	args.make_gray = true;
    args.resize_src = false;
    args.hog_width = 128;
    args.hog_height = 256;
    args.gr_threshold = group_threshold;
    args.hit_threshold = hit_threshold;
    args.n_orients = 9;
    args.bin_size = 8;
    args.overlap_threshold=0.9;
    args.p_accept = 0.99;
    args.lambda = 0.1;
    args.epsilon= 0.099;
    args.tolerance = 1e-1;
    args.n_iterations = 1e2;
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    if(USE_COLOR){
    	int channels = 3;
    	this->n_descriptors=(args.n_orients*3+5-1)*(args.hog_width/args.bin_size)*(args.hog_height/args.bin_size) + (this->args.hog_width/2)*(this->args.hog_height/2)*channels;
    }
    else this->n_descriptors = (args.n_orients*3+5-1)*(args.hog_width/args.bin_size)*(args.hog_height/args.bin_size);
    //else this->n_descriptors = args.hog_width/8 * args.hog_height/8 * (3*args.nbins+5);
    this->generator.seed(seed1);
    this->feature_values=MatrixXd::Zero(0,this->n_descriptors);
	this->labels.resize(0);
	this->num_frame=0;
	this->max_value=1.0;
	this->dataClean();
	this->initialized=false;
}


vector<Rect> CPU_LR_HOGDetector::detect(Mat &frame, vector<Rect> &detections, vector<double> &weights, MatrixXd &features)
{
	Mat current_frame;
	frame.copyTo(current_frame);
	// set input image on which we will run segmentation
    vector<Rect> samples,raw_detections;
    this->detections.clear();
	setUseOptimized(true);
    setNumThreads(4);
    Ptr<SelectiveSearchSegmentation> ss = createSelectiveSearchSegmentation();
    ss->setBaseImage(current_frame);
    ss->switchToSelectiveSearchFast();
    ss->process(samples);
    //this->feature_values=MatrixXd::Zero(samples.size(),this->n_descriptors); //
	this->weights.clear();
	double max_prob = 0.0;
	//cout << "Region Proposals: "   << samples.size() << endl;
	MatrixXd temp_features_matrix = MatrixXd::Zero(samples.size(),this->n_descriptors);
	for(size_t i = 0; i < samples.size(); i++){
		Rect current_window = samples[i];
		Mat subImage = current_frame(current_window);
		VectorXd hogFeatures = this->genHog(subImage);
		VectorXd temp;
		if(USE_COLOR){
			VectorXd rawPixelsFeatures = this->genRawPixels(subImage);
			temp.resize(hogFeatures.rows() + rawPixelsFeatures.rows());
			temp << hogFeatures, rawPixelsFeatures;
		}
		else{
			temp.resize(hogFeatures.rows());
			temp << hogFeatures;
		}
		//temp.normalize();
		temp_features_matrix.row(i) = temp;	
	}
	VectorXd dataNorm = temp_features_matrix.rowwise().squaredNorm().array().sqrt();
	temp_features_matrix = temp_features_matrix.array().colwise() / dataNorm.array();
	VectorXd predict_prob = this->logistic_regression.predict(temp_features_matrix, true);
	max_prob = predict_prob.maxCoeff();
	for (int i = 0; i < predict_prob.rows(); ++i)
	{
		if(predict_prob(i) > args.hit_threshold){
			Rect current_window = samples[i];
			//stringstream ss;
			//ss << predict_prob(i);
			//max_prob=MAX(max_prob,predict_prob(i));
			this->weights.push_back(predict_prob(i));
			raw_detections.push_back(current_window);
		}
	}
	if(this->args.gr_threshold > 0) {
		nms2(raw_detections, this->weights, detections, weights, args.gr_threshold, 2);
	}
	else{
		detections = raw_detections;
	}

	features = MatrixXd(detections.size(), this->n_descriptors);
	for(size_t i = 0; i < detections.size(); i++){
		Rect current_window = detections[i];
		Mat subImage = current_frame(current_window);
		features.row(i) = this->genHog(subImage);
	}

	this->num_frame++;
	return this->detections;
}




void CPU_LR_HOGDetector::train()
{
	if(!this->initialized) {
		this->logistic_regression.init(this->feature_values, this->labels, args.lambda, true,true,true);
		cout << "init train!" << endl;
		tools.writeToCSVfile("Model_means.csv", this->logistic_regression.featureMean.transpose());
		tools.writeToCSVfile("Model_stds.csv", this->logistic_regression.featureStd.transpose());
		tools.writeToCSVfile("Model_maxs.csv", this->logistic_regression.featureMax.transpose());
		tools.writeToCSVfile("Model_mins.csv", this->logistic_regression.featureMin.transpose());
		
	}
	this->initialized=true;
	this->logistic_regression.train(args.n_iterations, args.epsilon, args.tolerance);
	VectorXd weights = this->logistic_regression.getWeights();
	VectorXd bias(1);
	bias << this->logistic_regression.getBias();
	tools.writeToCSVfile("Model_weights.csv", weights);
	tools.writeToCSVfile("Model_bias.csv", bias);
}

VectorXd CPU_LR_HOGDetector::getFeatures(Mat &frame){
	return this->genHog(frame);
}

vector<double> CPU_LR_HOGDetector::getWeights(){
	return this->weights;
}

VectorXd CPU_LR_HOGDetector::genHog(Mat &frame)
{	
	int interpolation;
	if(args.hog_width > frame.size().height){
        interpolation = INTER_LINEAR;
      }else{
        interpolation = INTER_AREA;
    }
	Mat current_frame;
	resize(frame,current_frame,Size(args.hog_width, args.hog_height),0,0,interpolation);
	cv::cvtColor(current_frame, current_frame, CV_BGR2GRAY);
	current_frame.convertTo(current_frame, CV_32FC1);
	//extract(const cv::Mat & img, int use_hog = 2, int bin_size = 4, int n_orients = 9, int soft_bin = -1, float clip = 0.2)
	vector<Mat> mat_hog_features=FHoG::extract(current_frame, 2, args.bin_size, args.n_orients);
	int hog_channels=mat_hog_features.size();
	int cidx=0;
	VectorXd hog_features=VectorXd::Zero((args.n_orients*3+5-1)*(args.hog_width/args.bin_size)*(args.hog_height/args.bin_size));
	for (int ch = 0; ch < hog_channels; ++ch){
		for (int i = 0; i < mat_hog_features[ch].rows; i++){   
			for (int j = 0; j < mat_hog_features[ch].cols; j++){
				hog_features(cidx) =mat_hog_features[ch].at<float>(i,j);
				cidx++;
			}
		}
	}
	hog_features.normalize();
	return hog_features;
}

VectorXd CPU_LR_HOGDetector::genRawPixels(Mat &frame)
{
  int interpolation;
  if(args.hog_width/2 > frame.size().height){
        interpolation = INTER_LINEAR;
      }else{
        interpolation = INTER_AREA;
    }
  Mat current_frame;
  frame.copyTo(current_frame);
  resize(current_frame,current_frame,Size(args.hog_width/2, args.hog_height/2),0,0,interpolation);
  cvtColor(current_frame, current_frame, COLOR_BGR2Lab);
  current_frame.convertTo( current_frame, CV_32FC1, 1. / 255., -0.5); //to double
  Mat ch1(current_frame.size(), CV_32FC1);
  Mat ch2(current_frame.size(), CV_32FC1);
  Mat ch3(current_frame.size(), CV_32FC1);
  vector<Mat> color_features = {ch1, ch2, ch3};
  cv::split(current_frame, color_features);
  split(current_frame, color_features);
  //vector<Mat> cn_feat = CNFeat::extract(current_frame);
  int channels = color_features.size();
  //cout << channels << "," << cn_feat[0].cols << "," << cn_feat[0].rows << endl;
  VectorXd rawPixelsFeatures(color_features[0].cols*color_features[0].rows*channels);
  int cidx=0;
  for (int ch = 0; ch < channels; ++ch){   
      for(int c = 0; c < color_features[ch].cols ; c++){
        for(int r = 0; r < color_features[ch].rows ; r++){
            rawPixelsFeatures(cidx) = (double)color_features[ch].at<float>(r,c);
            cidx++;
        }
    }
  }
  double normTerm = rawPixelsFeatures.norm();
  if (normTerm > 1e-6){
    rawPixelsFeatures.normalize();
  }
  return rawPixelsFeatures;
}



void CPU_LR_HOGDetector::loadModel(VectorXd weights,VectorXd featureMean, VectorXd featureStd, VectorXd featureMax, VectorXd featureMin, double bias){
	this->logistic_regression.init(true, true, true);
	this->logistic_regression.setWeights(weights);
	this->logistic_regression.setBias(bias);
	this->logistic_regression.featureMean = featureMean;
	this->logistic_regression.featureStd = featureStd;
	this->logistic_regression.featureMax = featureMax;
	this->logistic_regression.featureMin = featureMin;
	this->initialized=true;
}

void CPU_LR_HOGDetector::generateFeatures(Mat &frame, double label)
{	
	// set input image on which we will run segmentation
   vector<Rect> samples;
    if(frame.cols > args.hog_width && frame.rows> args.hog_height){
    	setUseOptimized(true);
    	setNumThreads(4);
    	Ptr<SelectiveSearchSegmentation> ss = createSelectiveSearchSegmentation();
    	ss->setBaseImage(frame);
    	ss->switchToSelectiveSearchFast();
    	ss->process(samples);
    }
    else{
    	Rect centerROI(0, 0, args.hog_width,args.hog_height);
    	samples.push_back(centerROI);
    }
    for(unsigned i=0;i<samples.size();i++){
    	Rect current_window=samples[i];
		Mat subImage = frame(current_window);
		VectorXd hogFeatures = this->getFeatures(subImage);	
		//cout << "hog vector : " <<  hogFeatures.size() << "," << this->feature_values.rows() << "," << this->feature_values.cols() << endl;
		this->feature_values.conservativeResize(this->feature_values.rows() + 1, NoChange);
		this->feature_values.row(this->feature_values.rows()-1)=hogFeatures;
    }
    this->labels = VectorXd::Constant(samples.size(),label);
}

void CPU_LR_HOGDetector::dataClean(){
	this->feature_values.resize(0,this->n_descriptors);
	this->labels.resize(0);
}

void CPU_LR_HOGDetector::draw()
{
	for (size_t i = 0; i < this->detections.size(); i++)
    {
        Rect r = this->detections[i];
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.8);
        rectangle(this->frame, r.tl(), r.br(), cv::Scalar(255,0,0), 3);
    }
}


void CPU_LR_HOGDetector::saveToCSV(string name, bool append){
	tools.writeToCSVfile(name+"_values.csv", this->feature_values, append);
	tools.writeToCSVfile(name+"_labels.csv", this->labels, append);
}

void CPU_LR_HOGDetector::loadFeatures(MatrixXd features, VectorXd labels){
	this->dataClean();
	this->feature_values = features;
	this->labels = labels;
	if(this->initialized) this->logistic_regression.setData(this->feature_values,this->labels);	
}

VectorXd CPU_LR_HOGDetector::predictTest(MatrixXd features,bool data_processing){
	return this->logistic_regression.predict(features,false,data_processing);
}