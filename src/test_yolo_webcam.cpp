#include "test_yolo_webcam.hpp"

TestYOLOWebcam::TestYOLOWebcam(){}

TestYOLOWebcam::TestYOLOWebcam(string model_cfg, string model_binary, string class_names, float min_confidence,int cameraDevice){
    this->model_cfg = model_cfg;
    this->model_binary = model_binary;
    this->class_names = class_names;
    this->min_confidence = min_confidence;
    this->cap = VideoCapture(cameraDevice);
    if(!this->cap.isOpened()){
        cout << "Couldn't find camera: " << cameraDevice << endl;
    }
}

void TestYOLOWebcam::video_capture_buffer(queue<Mat> &frame_buffer){
    Mat frame;
    for (;;)
    {
        cap.read(frame);
        if (frame.empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }
        else{
            frame_buffer.push(frame);
        }
    }
}

void TestYOLOWebcam::run(queue<Mat> &frame_buffer){
    bool verbose=true;
	PHDGaussianMixture filter(verbose);
    if(verbose) namedWindow("YOLO Detector", WINDOW_NORMAL);

    YOLODetector detector(this->model_cfg, this->model_binary, this->class_names, this->min_confidence);
    Mat frame;
    for(;;)
    {

        if (!frame_buffer.empty()) {
            frame=frame_buffer.back();
            frame_buffer.pop();
            vector<Rect> detections = detector.detect(frame);
            std::vector<MyTarget> estimates;
            

            if (!filter.is_initialized())
            {
                filter.initialize(frame, detections, detector.weights);
                estimates = filter.estimate(frame, true);
                //filter.draw_particles(frame, Scalar(255, 255, 255));
            }
            else
            {
                filter.predict();
                filter.update(frame, detections, detector.weights);
                estimates = filter.estimate(frame, true);
                //filter.draw_particles(frame, Scalar(255, 255, 255));
            }
            /*for(size_t j = 0; j < estimates.size(); j++){
                cout << estimates.at(j).label
                << "," << estimates.at(j).bbox.x
                << "," << estimates.at(j).bbox.y
                << "," << estimates.at(j).bbox.width
                << "," << estimates.at(j).bbox.height
                << ",1,-1,-1,-1" << endl;
            }*/
            //detector.draw(frame);
            imshow("YOLO Detector", frame);
        } 
        
		if (waitKey(1) >= 0) break;
    }
}

int main(int argc, char const *argv[]){
    string first_frame_file, ground_truth_filename, model_cfg, model_binary, class_names;
    float min_confidence;
    int video;

    if(argc != 11)
    {
		cout << "Incorrect input list" << endl;
		cout << "exiting..." << endl;
		return EXIT_FAILURE;
    }
    else
    {
        if(strcmp(argv[1], "-config") == 0){
            model_cfg = argv[2];
        }
        else
        {
            cout << "No model configuration given" << endl;
            cout << "exiting..." << endl;
            return EXIT_FAILURE;
        }
        if(strcmp(argv[3], "-model") == 0){
            model_binary = argv[4];
        }
        else
        {
            cout << "No model given" << endl;
            cout << "exiting..." << endl;
            return EXIT_FAILURE;
        }
        if(strcmp(argv[5], "-classes") == 0){
            class_names = argv[6];
        }
        else
        {
            cout << "No class names given" << endl;
            cout << "exiting..." << endl;
            return EXIT_FAILURE;
        }
        if(strcmp(argv[7], "-min_confidence") == 0){
            min_confidence = stod(argv[8]);
        }
        else
        {
            cout << "No min confidence given" << endl;
            cout << "exiting..." << endl;
            return EXIT_FAILURE;
        }
        if (strcmp(argv[9], "-video") == 0)
	  	{
	  		video = stoi(argv[10]);
		}
        queue<Mat> frame_buffer;
        TestYOLOWebcam detector(model_cfg, model_binary, class_names, min_confidence,video);
        thread th1(&TestYOLOWebcam::video_capture_buffer,detector,ref(frame_buffer)); 
        thread th2(&TestYOLOWebcam::run,detector,ref(frame_buffer)); 
        //detector.run(true,frame_buffer);
        th1.join();
        th2.join();
    }
}