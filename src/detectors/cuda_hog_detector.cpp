#include "cuda_hog_detector.hpp"


CUDA_HOGDetector::CUDA_HOGDetector(int group_threshold, double hit_threshold){
	this->gpu_hog = cuda::HOG::create();
	this->group_threshold = group_threshold;
	this->hit_threshold = hit_threshold;
	Mat detector = this->gpu_hog->getDefaultPeopleDetector();
	this->gpu_hog->setSVMDetector(detector);
}

vector<Rect> CUDA_HOGDetector::detect(Mat &frame)
{
    cuda::GpuMat gpu_img;
    this->detections.clear();
    vector<double> weights;
    Mat img_aux;
    cvtColor(frame, img_aux, COLOR_BGR2BGRA);
    gpu_img.upload(img_aux);
    this->gpu_hog->setHitThreshold(this->hit_threshold);
    this->gpu_hog->setGroupThreshold(this->group_threshold);
    this->gpu_hog->detectMultiScale(gpu_img, this->detections,&weights);
	double* ptr = &weights[0];
	this->weights = Eigen::Map<Eigen::VectorXd>(ptr, weights.size());
	this->frame = frame;
	return this->detections;
}

void CUDA_HOGDetector::draw()
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

MatrixXd CUDA_HOGDetector::getFeatureValues()
{
	MatrixXd hogFeatures(this->detections.size(), this->gpu_hog->getDescriptorSize());
	//this->gpu_hog.winSize = Size(64,128);
	//this->hog.nbins = 32;
	Mat subImage,hog_descriptor;
	vector<float> features;
	cuda::GpuMat gpu_img,hog_img;
	for (size_t i = 0; i < this->detections.size(); ++i)
	{
		subImage = this->frame(this->detections.at(i));
		if(this->frame.rows > 0 && this->frame.cols > 0)
		{
			resize(subImage, subImage,  Size(64,128), 0, 0, INTER_LINEAR);
			cvtColor(subImage, subImage, COLOR_BGR2BGRA);
			gpu_img.upload(subImage);
			this->gpu_hog->compute(gpu_img, hog_img);
			hog_img.download(hog_descriptor);
			for (size_t j = 0; j < hog_descriptor.cols; ++j)
			{
				hogFeatures(i,j) = hog_descriptor.at<double>(0,j);
			}
		}
	}
	hogFeatures.normalize();
	return hogFeatures;
}

VectorXd CUDA_HOGDetector::getDetectionWeights(){
	this->weights.normalize();
	return this->weights;
}