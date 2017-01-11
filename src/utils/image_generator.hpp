#ifndef IMAGE_GENERATOR_H
#define IMAGE_GENERATOR_H

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

using namespace std;
using namespace cv;

typedef struct{
	int label;
	Rect bbox;
}Target;

class ImageGenerator{

public:
	ImageGenerator();
	ImageGenerator(string _firstFrameFileName, string _groundTruthFileName);
	unsigned int getDatasetSize();
	Mat getFrame(unsigned int frameIndex);
	vector<Target> getGroundTruth(unsigned int frameIndex);

private:
	//unsigned int frameId;
	vector<Mat> images;
	vector<vector<Target>> groundTruth;
	string frameFileName, groundTruthFileName;
	void getNextFilename(string& fn);
	void readGroundTruth();

};

#endif