#include "image_generator.hpp"

ImageGenerator::ImageGenerator(){
}

ImageGenerator::ImageGenerator(string _firstFrameFileName, string _groundTruthFileName){
	this->frameFileName = _firstFrameFileName;
	this->groundTruthFileName = _groundTruthFileName;
	//this->frameId = 0;

	Mat currentFrame = imread(this->frameFileName);
	this->images.push_back(currentFrame);
	while(1){
		
		getNextFilename(this->frameFileName);
	    currentFrame = imread(this->frameFileName);
	    
	    if(currentFrame.empty()){
	      	break;
	    }
	    else{
	    	this->images.push_back(currentFrame);
	    }
	}
	
	readGroundTruth();
}

void ImageGenerator::getNextFilename(string& fn){
    size_t index = fn.find_last_of("/");
    if(index == string::npos) {
        index = fn.find_last_of("\\");
    }
    size_t index2 = fn.find_last_of(".");
    string prefix = fn.substr(0,index+1);
    string suffix = fn.substr(index2);
    string frameNumberString = fn.substr(index+1, index2-index-1);
    istringstream iss(frameNumberString);
    int frameNumber = 0;
    iss >> frameNumber;
    ostringstream oss;
    oss << (frameNumber + 1);
    
    string nextFrameNumberString = oss.str();
    string nextFrameFilename = prefix +nextFrameNumberString + suffix;
    fn.assign(nextFrameFilename);
}

void ImageGenerator::readGroundTruth(){
	ifstream gt_file(this->groundTruthFileName, ios::in);
	string line;
	this->groundTruth.resize(getDatasetSize());

	while(getline(gt_file, line)){
		Target target;
		Rect bbox;
		
		//cout << "line: " << line << endl;

		int pos1 = 0;
		int pos2 = line.find(" ");
		int frameNumber = stoi(line.substr(pos1,pos2)) - 1;

		pos1 = pos2;
		pos2 = line.find(" ",pos1 + 1);
		bbox.x = stod(line.substr(pos1 + 1,pos2-pos1-1));

		//cout << "bbox.x: " << bbox.x << endl;
		//cout << "pos1: " << pos1 << "\tpos2: " << pos2 << endl;
		
		pos1 = pos2;
		pos2 = line.find(" ",pos1 + 1);
		bbox.y = stod(line.substr(pos1 + 1,pos2-pos1-1));

		//cout << "bbox.y: " << bbox.y << endl;
		//cout << "pos1: " << pos1 << "\tpos2: " << pos2 << endl;

		pos1 = pos2;
		pos2 = line.find(" ",pos1 + 1);
		bbox.width = stod(line.substr(pos1 + 1,pos2-pos1-1));

		//cout << "bbox.width: " << bbox.width << endl;
		//cout << "pos1: " << pos1 << "\tpos2: " << pos2 << endl;

		pos1 = pos2;
		pos2 = line.find(" ",pos1 + 1);
		bbox.height = stod(line.substr(pos1 + 1,pos2-pos1-1));

		//cout << "bbox.height: " << bbox.height << endl;
		//cout << "pos1: " << pos1 << "\tpos2: " << pos2 << endl;

		//cout << "DatasetSize: " << getDatasetSize() << endl;
		target.bbox = bbox;
		this->groundTruth[frameNumber].push_back(target);
	}
	//cout << "groundTruth size: " << this->groundTruth.size() << endl;
}

unsigned int ImageGenerator::getDatasetSize(){
	return images.size();
}

vector<Target> ImageGenerator::getGroundTruth(unsigned int frameIndex){
	return this->groundTruth[frameIndex];
}

Mat ImageGenerator::getFrame(unsigned int frameIndex){
	Mat currentFrame = images[frameIndex].clone();
	return currentFrame;
}