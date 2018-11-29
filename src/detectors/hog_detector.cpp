#include "hog_detector.hpp"

/*#ifndef PARAMS
const int GROUP_THRESHOLD = 0.5;
const double HIT_THRESHOLD = 0.1;
#endif*/
HOGDetector::HOGDetector(){}

HOGDetector::HOGDetector(int group_threshold, double hit_threshold){
	this->hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
	this->group_threshold = group_threshold;
	this->hit_threshold = hit_threshold;
}

vector<Rect> HOGDetector::detect(Mat &frame)
{
    this->detections.clear();
    vector<double> weights;
	this->hog.detectMultiScale(frame, this->detections, weights, this->hit_threshold, Size(8,8), Size(32,32), 1.05, this->group_threshold);
	double* ptr = &weights[0];
	this->weights = Eigen::Map<Eigen::VectorXd>(ptr, weights.size());
	this->frame = frame;
	return this->detections;
}

void HOGDetector::draw()
{
	for (size_t i = 0; i < this->detections.size(); i++)
    {
        Rect r = this->detections[i];

        // The HOG detector returns slightly larger rectangles than the real objects,
        // so we slightly shrink the rectangles to get a nicer output.
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.8);
        rectangle(this->frame, r.tl(), r.br(), cv::Scalar(255,0,0), 3);
    }
}

MatrixXd HOGDetector::getFeatureValues(Mat &frame, vector<MyTarget> detections)
{
	MatrixXd hogFeatures(detections.size(), 3780);
	this->hog.winSize = Size(64,128);
	//this->hog.nbins = 32;
	Mat subImage;
	vector<float> features;

	for (size_t i = 0; i < detections.size(); ++i)
	{
		subImage = frame(detections.at(i).bbox);
		if(frame.rows > 0 && frame.cols > 0)
		{
			resize(subImage, subImage, this->hog.winSize, 0, 0, INTER_LINEAR);
			cvtColor(subImage, subImage, COLOR_RGB2GRAY);
			this->hog.compute(subImage, features, Size(0,0), Size(0,0));
			for (size_t j = 0; j < features.size(); ++j)
			{
				hogFeatures(i,j) = features.at(j);
			}
		}
	}
	
	hogFeatures.normalize();
	
	return hogFeatures;
}

VectorXd HOGDetector::getDetectionWeights(){
	this->weights.normalize();
	return this->weights;
}