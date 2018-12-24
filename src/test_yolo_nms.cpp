#include "test_yolo.hpp"

TestYOLODetector::TestYOLODetector(){}

TestYOLODetector::TestYOLODetector(string first_frame_file, string ground_truth_filename, 
    string model_cfg, string model_binary, string class_names, float min_confidence){
    this->first_frame_file = first_frame_file;
    this->ground_truth_filename = ground_truth_filename;
    this->model_cfg = model_cfg;
    this->model_binary = model_binary;
    this->class_names = class_names;
    this->min_confidence = min_confidence;
    this->generator = ImageGenerator(this->first_frame_file, this->ground_truth_filename);
}

void TestYOLODetector::run(bool verbose, float threshold){
	//PHDGaussianMixture filter(verbose, dpp_epsilon);
    //PHDGaussianMixture filter(verbose);
    PHDGaussianMixture filter(threshold, 0.7,1, 0.1);
    if(verbose) namedWindow("YOLO Detector", WINDOW_NORMAL);

    YOLODetector detector(this->model_cfg, this->model_binary, this->class_names, this->min_confidence);

    for(size_t i = 0; i < this->generator.getDatasetSize(); i++)
    {
        Mat frame = this->generator.getFrame(i);
        vector<Rect> detections = detector.detect(frame);
        //vector<Target> gt = this->generator.getGroundTruth(i);
        //detector.draw(frame);
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
        for(size_t j = 0; j < estimates.size(); j++){
			cout << i + 1
			<< "," << estimates.at(j).label
			<< "," << estimates.at(j).bbox.x
			<< "," << estimates.at(j).bbox.y
			<< "," << estimates.at(j).bbox.width
			<< "," << estimates.at(j).bbox.height
			<< ",1,-1,-1,-1" << endl;
		}
        //detector.draw(frame);
        
        if (verbose) {
            imshow("YOLO Detector", frame);
		    waitKey(1);
        }
        
    }
}

int main(int argc, char const *argv[]){
    string first_frame_file, ground_truth_filename, model_cfg, model_binary, class_names;
    float min_confidence, nms_epsilon;
    bool verbose;

    if(argc != 17)
    {
		cout << "Incorrect input list" << endl;
		cout << "exiting..." << endl;
		return EXIT_FAILURE;
    }
    else
    {
        if(strcmp(argv[1], "-img") == 0){
            first_frame_file = argv[2];
        }
        else
        {
            cout << "No images given" << endl;
            cout << "exiting..." << endl;
            return EXIT_FAILURE;
        }
        if(strcmp(argv[3], "-gt") == 0){
            ground_truth_filename = argv[4];
        }
        else
        {
            cout << "No groundtruth given" << endl;
            cout << "exiting..." << endl;
            return EXIT_FAILURE;
        }
        if(strcmp(argv[5], "-config") == 0){
            model_cfg = argv[6];
        }
        else
        {
            cout << "No model configuration given" << endl;
            cout << "exiting..." << endl;
            return EXIT_FAILURE;
        }
        if(strcmp(argv[7], "-model") == 0){
            model_binary = argv[8];
        }
        else
        {
            cout << "No model given" << endl;
            cout << "exiting..." << endl;
            return EXIT_FAILURE;
        }
        if(strcmp(argv[9], "-classes") == 0){
            class_names = argv[10];
        }
        else
        {
            cout << "No class names given" << endl;
            cout << "exiting..." << endl;
            return EXIT_FAILURE;
        }
        if(strcmp(argv[11], "-min_confidence") == 0){
            min_confidence = stod(argv[12]);
        }
        else
        {
            cout << "No min confidence given" << endl;
            cout << "exiting..." << endl;
            return EXIT_FAILURE;
        }

        if(strcmp(argv[13], "-epsilon") == 0){
            nms_epsilon = stod(argv[14]);
        }
        else{
            cout << "No dpp epsilon value given" << endl;
            cout << "exiting..." << endl;
            return EXIT_FAILURE;
		}
        if (strcmp(argv[15], "-verbose") == 0)
        {
            verbose = (stoi(argv[16]) == 1) ? true : false;
        }
        TestYOLODetector detector(first_frame_file, ground_truth_filename, model_cfg, model_binary, class_names, min_confidence);
        detector.run(verbose, nms_epsilon);
    }
}