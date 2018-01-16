#include "image_generator.hpp"

#ifndef PARAMS
  const int FEATURES_NUM = 15872;
#endif

using namespace std;
using namespace cv;

ImageGenerator::ImageGenerator(){
}

ImageGenerator::ImageGenerator(string _firstFrameFilename, string _groundTruthFile){
  this->frame_id = 0;
  string FrameFilename, gtFilename;
  FrameFilename = _firstFrameFilename;
  gtFilename = _groundTruthFile;
  Mat current_frame = imread(FrameFilename);
  this->images.push_back(current_frame);
  while(1){
    getNextFilename(FrameFilename);
    current_frame = imread(FrameFilename );
    if(current_frame.empty()){
      break;
    }
    else{
      this->images.push_back(current_frame);
    }
  }
  readGroundTruth(_groundTruthFile);
}

ImageGenerator::ImageGenerator(string _firstFrameFilename, string _groundTruthFile, string _detectionsFile){
  this->frame_id = 0;
  string FrameFilename, gtFilename, detFilename;
  FrameFilename = _firstFrameFilename;
  gtFilename = _groundTruthFile;
  Mat current_frame = imread(FrameFilename);
  this->images.push_back(current_frame);
  while(1){
    getNextFilename(FrameFilename);
    current_frame = imread(FrameFilename);
    if(current_frame.empty()){
      break;
    }
    else{
      this->images.push_back(current_frame);
    }
  }
  readDetections(_detectionsFile);
  readGroundTruth(_groundTruthFile);
}

Mat ImageGenerator::getFrame(int frame_num){
  Mat current_frame = this->images[frame_num].clone();
  return current_frame;
}

vector<Rect> ImageGenerator::getDetections(int frame_num){
  return this->detections[frame_num];
}

VectorXd ImageGenerator::getDetectionWeights(int frame_num){
  return this->detection_weights[frame_num];
}

vector<Target> ImageGenerator::getGroundTruth(int frame_num){
  return this->ground_truth[frame_num];
}

MatrixXd ImageGenerator::getDetectionFeatures(int frame_num){
  return this->features[frame_num];
}

bool ImageGenerator::hasEnded(){
  if(frame_id >= (int) this->images.size()){
    return true;
  }else{
    return false;
  }
}

void ImageGenerator::moveNext(){
  this->frame_id++;
}

size_t ImageGenerator::getDatasetSize(){
  return this->images.size();
}

void ImageGenerator::getNextFilename(string& fn){
    size_t index = fn.find_last_of("/");
    if(index == string::npos) {
        index = fn.find_last_of("\\");
    }
    size_t index2 = fn.find_last_of(".");
    string prefix = fn.substr(0,index+1);
    string suffix = fn.substr(index2);
    string frameNumberString = fn.substr(index + 1, index2 - index - 1);
    istringstream iss(frameNumberString);
    int frameNumber = 0;
    iss >> frameNumber;
    ostringstream oss;
    oss << (frameNumber + 1);
    string zeros("0000000");
    string nextFrameNumberString = oss.str();
    string nextFrameFilename = prefix + zeros.substr(0, zeros.length() - 1 - nextFrameNumberString.length()) + nextFrameNumberString + suffix;
    fn.assign(nextFrameFilename);
}

void ImageGenerator::readDetections(string detFilename){
  ifstream dt_file(detFilename.c_str(), ios::in);
  string line;
  this->detections.resize(getDatasetSize());
  this->detection_weights.resize(getDatasetSize());
  this->features.resize(getDatasetSize());

  vector<double> coords(4,0);
  int frame_num;
  
  VectorXd row(FEATURES_NUM);

  while (getline(dt_file, line)) {
    Rect rect;
    size_t pos2 = line.find(",");
    size_t pos1 = 0;
    if(pos2>pos1){
      frame_num=stoi(line.substr(pos1, pos2)) - 1;
      pos1 = line.find(",",pos2 + 1);
      pos2 = line.find(",",pos1 + 1);
      coords[0] = stoi(line.substr(pos1 + 1, pos2 - pos1 - 1));
      for(int j = 1; j < 4; j++){
        pos1 = pos2;
        pos2 = line.find(",", pos1 + 1);
        coords[j] = stoi(line.substr(pos1 + 1,pos2 - pos1 - 1));
      }
      rect.x = coords[0];
      rect.y = coords[1];
      rect.width = coords[2];
      rect.height = coords[3];
      this->detections[frame_num].push_back(rect);
      
      pos1 = pos2;
      pos2 = line.find(",", pos1 + 1);
      this->detection_weights[frame_num].conservativeResize( this->detection_weights[frame_num].size() + 1 );
      this->detection_weights[frame_num](this->detection_weights[frame_num].size() - 1) = stod(line.substr(pos1 + 1, pos2 - pos1 - 1));
  
      for(int j = 1; j < 3; j++){
        pos1 = pos2;
        pos2 = line.find(",", pos1 + 1);
      }
  
      for(int j = 1; j < FEATURES_NUM; j++){
        pos1 = pos2;
        pos2 = line.find(",", pos1 + 1);
        row(j) = stod(line.substr(pos1 + 1, pos2 - pos1 - 1));
      }
      this->features[frame_num].conservativeResize(this->features[frame_num].rows() + 1, FEATURES_NUM);
      this->features[frame_num].row(this->features[frame_num].rows() - 1 ) = row; 
    }
  }
}

void ImageGenerator::readGroundTruth(string gtFilename){
  ifstream gt_file(gtFilename.c_str(), ios::in);
  string line;
  this->ground_truth.resize(getDatasetSize());
  vector<double> coords(4,0);
  int frame_num;
  
  while (getline(gt_file, line)) {
    Target target; 
    Rect rect;
    size_t pos2 = line.find(",");
    size_t pos1 = 0;
    frame_num = stoi(line.substr(pos1, pos2)) - 1;
    pos1 = pos2;
    pos2 = line.find(",",pos1 + 1);
    target.label = stoi(line.substr(pos1 + 1, pos2 - pos1 - 1));   
    pos1 = pos2;
    pos2 = line.find(",", pos1 + 1);
    coords[0] = stod(line.substr(pos1 + 1, pos2 - pos1 - 1));
    for(int j = 1; j < 4; j++){
      pos1 = pos2;
      pos2 = line.find(",", pos1 + 1);
      coords[j] = stod(line.substr(pos1 + 1, pos2 - pos1 - 1));  
    }
    rect.x = coords[0];
    rect.y = coords[1];
    rect.width = coords[2];
    rect.height = coords[3];
    target.bbox = rect;
    
    /*pos1 = pos2;
    pos2 = line.find(",", pos1 + 1);
    target.conf = stod(line.substr(pos1 + 1, pos2 - pos1 - 1));*/
    
    this->ground_truth[frame_num].push_back(target);
    /*cout << frame_num
    << "," << target.label
    << "," << target.bbox.x
    << "," << target.bbox.y
    << "," << target.bbox.width
    << "," << target.bbox.height
    << ",-1,-1,-1,-1" << endl;*/

  }
}