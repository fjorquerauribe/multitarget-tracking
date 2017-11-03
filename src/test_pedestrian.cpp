#include "test_pedestrian.hpp"

TestPedestrian::TestPedestrian(){}

TestPedestrian::TestPedestrian(int _npart)
{
	this->npart = _npart;
}

void TestPedestrian::run()
{
	namedWindow("PHD Filter", WINDOW_NORMAL);//WINDOW_NORMAL
	RNG rng( 0xFFFFFFFF );
	map<int,Scalar> color;
	PHDParticleFilter filter(this->npart);
    HOGDetector detector(0.5, 0.1);

	VideoCapture cap;
    // open the default camera, use something different from 0 otherwise;
    // Check VideoCapture documentation.
    if(!cap.open(0))
        return;
    for(;;)
    {
		Mat frame;
		cap >> frame;
		if( frame.empty() ) break; // end of video stream
		
		vector<Rect> preDetections = detector.detect(frame);
		vector<Target> estimates;

		if (!filter.is_initialized())
		{
			filter.initialize(frame, preDetections);
			filter.draw_particles(frame, Scalar(255, 255, 255));
			estimates = filter.estimate(frame, false);
		}
		else
		{
			filter.predict();
			filter.update(frame, preDetections);
			filter.draw_particles(frame, Scalar(255, 255, 255));
			estimates = filter.estimate(frame, false);
		}

		imshow("PHD Filter", frame);
		if( waitKey(10) == 27 ) break; // stop capturing by pressing ESC 
    }
    // the camera will be closed automatically upon exit
    // cap.close();
}

int main(int argc, char const *argv[])
{
	int _npart;
	if(argc != 3)
	{
		cout << "Incorrect input list" << endl;
		cout << "exiting..." << endl;
		return EXIT_FAILURE;
	}
	else
	{
	  	if (strcmp(argv[1], "-npart") == 0)
	  	{
	  		_npart = atoi(argv[2]);
	  	}
	  	else
	  	{
	  		cout << "No particles number given" << endl;
	  		cout << "exiting..." << endl;
	  		return EXIT_FAILURE;
	  	}
	  	TestPedestrian tracker(_npart);
	  	tracker.run();
	}
}