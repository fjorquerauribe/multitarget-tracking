#include "image_generator.hpp"

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
    getNextFilename(FrameFilename, "pets");
    current_frame = imread(FrameFilename );
    if(current_frame.empty()){
      break;
    }
    else{
      this->images.push_back(current_frame);
    }
  }
  readGroundTruth(_groundTruthFile, "pets");
  //cout << "images: " << getDatasetSize() << ", ground truth:" << this->ground_truth.size() << endl;
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
  //cout << "images: " << getDatasetSize() << ", detections:" << this->detections.size() << ", ground truth:" << this->ground_truth.size() << endl;
}

Mat ImageGenerator::getFrame(int frame_num){
  Mat current_frame = this->images[frame_num].clone();
  return current_frame;
}

vector<Rect> ImageGenerator::getDetections(int frame_num){
  return this->detections[frame_num];
}

vector<Target> ImageGenerator::getGroundTruth(int frame_num){
  return this->ground_truth[frame_num];
}

bool ImageGenerator::hasEnded(){
  if(frame_id >= (int) this->images.size()){
    return true;
  }else{
    return false;
  }
}

void ImageGenerator::moveNext(){
  cout << this->frame_id << endl;
  this->frame_id++;
}

size_t ImageGenerator::getDatasetSize(){
  return this->images.size();
}

void ImageGenerator::getNextFilename(string& fn, string dataset){
  
    size_t index = fn.find_last_of("/");
    if(index == string::npos) {
        index = fn.find_last_of("\\");
    }
    //size_t index1 = fn.find_last_of("0");
    size_t index2 = fn.find_last_of(".");
    string prefix = fn.substr(0,index+1);
    string suffix = fn.substr(index2);
    string frameNumberString = fn.substr(index + 1, index2 - index - 1);
    istringstream iss(frameNumberString);
    int frameNumber = 0;
    iss >> frameNumber;
    ostringstream oss;
    oss << (frameNumber + 1);
    //string zeros("0000000");
    string zeros("");
    if (!dataset.compare("mot")){
      zeros = "0000000";
    }
     
    string nextFrameNumberString = oss.str();
    string nextFrameFilename = prefix + zeros.substr(0, zeros.length() - 1 - nextFrameNumberString.length()) + nextFrameNumberString + suffix;
    fn.assign(nextFrameFilename);
}

void ImageGenerator::readDetections(string detFilename){
  ifstream dt_file(detFilename.c_str(), ios::in);
  string line;
  this->detections.resize(getDatasetSize());
  vector<double> coords(4,0);
  int frame_num;
  while (getline(dt_file, line)) {
    Rect rect;
    size_t pos2 = line.find(",");
    size_t pos1 = 0;
    frame_num=stoi(line.substr(pos1, pos2)) - 1;
    pos1 = line.find(",",pos2 + 1);
    pos2 = line.find(",",pos1 + 1);
    coords[0] = stoi(line.substr(pos1 + 1, pos2 - pos1 - 1));
    for(int j = 1; j < 4; j++){
      pos1 = pos2;
      pos2 = line.find(",", pos1 + 1);
      coords[j] = stoi(line.substr(pos1 + 1,pos2 - pos1 - 1));  
      //detections[atoi(frame_num)]=
    }
    rect.x = coords[0];
    rect.y = coords[1];
    rect.width = coords[2];
    rect.height = coords[3];
    //cout << frame_num << images.size() << endl;
    detections[frame_num].push_back(rect);  
  }
}

void ImageGenerator::readGroundTruth(string gtFilename, string dataset){
  ifstream gt_file(gtFilename.c_str(), ios::in);
  string line;
  ground_truth.resize(getDatasetSize());
  vector<double> coords(4,0);
  int frame_num;
  if (!dataset.compare("mot"))
  {
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
      ground_truth[frame_num].push_back(target);
    }
  }
  else{
    while (getline(gt_file, line)) {
      Target target; 
      Rect rect;
      size_t pos2 = line.find(" ");
      size_t pos1 = 0;
      frame_num = stoi(line.substr(pos1, pos2)) - 1;
      
      for(int j = 0; j < 4; j++){
        pos1 = pos2;
        pos2 = line.find(" ", pos1 + 1);
        coords[j] = stod(line.substr(pos1 + 1, pos2 - pos1 - 1));
      }

      rect.x = coords[0];
      rect.y = coords[1];
      rect.width = coords[2];
      rect.height = coords[3];

      target.bbox = rect;
      target.label = -1;
      ground_truth[frame_num].push_back(target);
    }
  }
  
}