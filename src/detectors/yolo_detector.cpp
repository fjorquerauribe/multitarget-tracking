#include "yolo_detector.hpp"

YOLODetector::YOLODetector(){
    this->initialized = false;
}

YOLODetector::YOLODetector(string model_cfg, string model_binary, string class_names, float min_confidence){
    this->net = readNetFromDarknet(model_cfg, model_binary);
    this->net.setPreferableBackend(DNN_BACKEND_OPENCV);
    this->net.setPreferableTarget(DNN_TARGET_CPU);
    this->min_confidence = min_confidence;

    ifstream class_names_file(class_names);
    
    if (class_names_file.is_open()) {
        string class_name = "";
        while(getline(class_names_file, class_name))
            this->class_names_vec.push_back(class_name);
    }
    this->initialized = true;
}

bool YOLODetector::is_initialized(){
    return this->initialized;
}

vector<Rect> YOLODetector::detect(Mat frame){
    this->detections.clear();

    if(frame.channels() == 4) cvtColor(frame, frame, COLOR_BGRA2BGR);
    Mat inputBlob = blobFromImage(frame, 1/255.F, Size(416,416), Scalar(), true, false);
    net.setInput(inputBlob, "data");
    Mat detectionMat;
    vector<Mat> outs;
    net.forward(outs, getOutputsNames(net));
    this->classes.clear();
    this->weights.resize(0);
    for (unsigned k=0; k<outs.size(); k++){
        detectionMat=outs[k];
        size_t j = 0;
        for(int i = 0; i < detectionMat.rows; i++){
        int prob_index = 5;
        int prob_size = detectionMat.cols - prob_index;
        float *prob_array_ptr = &detectionMat.at<float>(i, prob_index);
        size_t object_class = max_element(prob_array_ptr, prob_array_ptr + prob_size) - prob_array_ptr;
        float confidence = detectionMat.at<float>(i, (int)object_class + prob_index);
        int object_classes[4]={0,1,2,3};
        bool any_object=(int)object_class==object_classes[0]
                        ||  (int)object_class==object_classes[1]
                        ||  (int)object_class==object_classes[2]
                        ||  (int)object_class==object_classes[3];
        if (confidence > this->min_confidence &&  any_object) {
                Rect rect;
                float x_center = detectionMat.at<float>(i, 0) * frame.cols;
                float y_center = detectionMat.at<float>(i, 1) * frame.rows;
                float width = detectionMat.at<float>(i, 2) * frame.cols;
                float height = detectionMat.at<float>(i, 3) * frame.rows;
                
                Point p1(cvRound(x_center - width / 2), cvRound(y_center - height / 2));
                Point p2(cvRound(x_center + width / 2), cvRound(y_center + height / 2));
                this->detections.push_back(Rect(p1, p2));
                
                this->weights.conservativeResize(this->weights.size() + 1);
                this->weights(j) = confidence;
                j++;

                String class_name = object_class < this->class_names_vec.size() ? this->class_names_vec[object_class] : cv::format("unknown(%d)", object_class);
                this->classes.push_back(class_name);
            }
        }
    
    }
    return this->detections;
}

void YOLODetector::draw(Mat frame, Scalar color){
    Mat icon = imread("man-user.png",CV_LOAD_IMAGE_COLOR);
    for(size_t i = 0; i < this->detections.size(); i++)
    {
        Rect rect = this->detections[i];
        //String label = format("%s: %.2f", this->classes[i].c_str(), this->weights(i));
        String label = format("%s: %.2f", this->classes[i].c_str(), this->weights(i));
        int baseLine = 0;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        rectangle(frame, rect, color);
        rectangle(frame, Rect(rect.tl(), Size(rect.width, labelSize.height + baseLine)), color, FILLED);
        putText(frame, label, rect.tl() + Point(0, labelSize.height), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255));
        
    }
}

vector<String> YOLODetector::getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}