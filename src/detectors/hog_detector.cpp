#include "hog_detector.hpp"

HOGDetector::HOGDetector(){
	this->hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
}

vector<Rect> HOGDetector::detect(Mat &frame)
{
	// Run the detector with default parameters. to get a higher hit-rate
    // (and more false alarms, respectively), decrease the hitThreshold and
    // groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
    this->detections.clear();
    vector<double> foundWeights;
	this->hog.detectMultiScale(frame, this->detections, foundWeights, 0, Size(8,8), Size(32,32), 1.05, 0);
	

	// Cast foundWeights to Eigen Vector
	double* ptr = &foundWeights[0];
	this->foundWeights = Eigen::Map<Eigen::VectorXd>(ptr, foundWeights.size());


	/*cout << "foundWeights size: " << this->foundWeights.size() << endl;
	cout << "detections size: " << this->detections.size() << endl;
	cout << "foundWeights: " << this->foundWeights << endl;*/

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
    //cout << "detections size: " << this->detections.size() << endl;
}

MatrixXd HOGDetector::getFeatureValues()
{
	MatrixXd hogFeatures(this->detections.size(), 3780);
	this->hog.winSize = Size(64,128);
	//this->hog.nbins = 32;
	Mat subImage;
	vector<float> features;

	for (size_t i = 0; i < this->detections.size(); ++i)
	{
		subImage = this->frame(this->detections.at(i));
		if(this->frame.rows > 0 && this->frame.cols > 0)
		{
			resize(subImage, subImage, this->hog.winSize, 0, 0, INTER_LINEAR);
			cvtColor(subImage, subImage, COLOR_RGB2GRAY);
			this->hog.compute(subImage, features, Size(0,0), Size(0,0));
			for (size_t j = 0; j < features.size(); ++j)
			{
				//cout << "i:" << i << "\tj:" << j << "\tfeatures size:" << features.size() << endl;
				hogFeatures(i,j) = features.at(j);
			}
		}
	}

	return hogFeatures;
}

// hay que normalizar
VectorXd HOGDetector::getDetectionWeights(){
	return this->foundWeights;
}