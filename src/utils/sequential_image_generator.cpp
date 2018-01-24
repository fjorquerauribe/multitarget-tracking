#include "sequential_image_generator.hpp"

#ifndef PARAMS
  const int FEATURES_DIM = 128; // HOG:15872 | CNN:128 
#endif

using namespace std;
using namespace cv;

SequentialImageGenerator::SequentialImageGenerator(){
}

SequentialImageGenerator::SequentialImageGenerator(string _firstFrameFilename, string _groundTruthFile, string _detectionsFile){
  this->frame_id = 0;
  this->detectionsFileName = _detectionsFile;
  string FrameFilename, gtFilename;
  FrameFilename = _firstFrameFilename;
  gtFilename = _groundTruthFile;
  Mat current_frame = imread(FrameFilename);
  this->images.push_back(current_frame);

  this->dt_file = ifstream(this->detectionsFileName.c_str(), ios::in);

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
  
  readGroundTruth(_groundTruthFile);
}

Mat SequentialImageGenerator::getFrame(int frame_num){
  Mat current_frame = this->images[frame_num].clone();
  return current_frame;
}

vector<Rect> SequentialImageGenerator::getDetections(){
  return this->detections;
}

VectorXd SequentialImageGenerator::getDetectionWeights(){
  return this->detection_weights;
}

MatrixXd SequentialImageGenerator::getDetectionFeatures(){
  return this->features;
}

vector<Target> SequentialImageGenerator::getGroundTruth(int frame_num){
  return this->ground_truth[frame_num];
}

bool SequentialImageGenerator::hasEnded(){
  if(frame_id >= (int) this->images.size()){
    return true;
  }else{
    return false;
  }
}

void SequentialImageGenerator::moveNext(){
  this->frame_id++;
}

size_t SequentialImageGenerator::getDatasetSize(){
  return this->images.size();
}

void SequentialImageGenerator::getNextFilename(string& fn){
    size_t index = fn.find_last_of("/");
    if(index == string::npos) {
        index = fn.find_last_of("\\");
    }
    size_t index2 = fn.find_last_of(".");
    string prefix = fn.substr(0, index + 1);
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

void SequentialImageGenerator::readDetections(int frame){
  string line;

  this->detections.clear();
  this->detection_weights.resize(0);
  this->features.conservativeResize(0, FEATURES_DIM);
  int frame_num;
  vector<double> coords(4,0);
  VectorXd row(FEATURES_DIM);

  while(true){
    this->pointerPos = this->dt_file.tellg();

    getline(this->dt_file, line);

    Rect rect;
    size_t pos2 = line.find(",");
    size_t pos1 = 0;

    if(pos2 > pos1){
      frame_num = stoi(line.substr(pos1, pos2)) - 1;

      if(frame_num != frame){
        this->dt_file.seekg(this->pointerPos ,std::ios_base::beg);
        break;
      }

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

      this->detections.push_back(rect);

      pos1 = pos2;
      pos2 = line.find(",", pos1 + 1);
      this->detection_weights.conservativeResize( this->detection_weights.size() + 1 );
      this->detection_weights(this->detection_weights.size() - 1) = stod(line.substr(pos1 + 1, pos2 - pos1 - 1));
  
      for(int j = 0; j < 3; j++){
        pos1 = pos2;
        pos2 = line.find(",", pos1 + 1);
      }
  
      for(int j = 0; j < FEATURES_DIM; j++){
        pos1 = pos2;
        pos2 = line.find(",", pos1 + 1);
        row(j) = stod(line.substr(pos1 + 1, pos2 - pos1 - 1));
      }
      this->features.conservativeResize(this->features.rows() + 1, FEATURES_DIM);
      this->features.row(this->features.rows() - 1 ) = row;

    }
  }
}

void SequentialImageGenerator::readGroundTruth(string gtFilename){
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
