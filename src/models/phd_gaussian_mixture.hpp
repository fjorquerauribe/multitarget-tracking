#ifndef PHD_PARTICLE_FILTER
#define PHD_PARTICLE_FILTER

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include "../likelihood/gaussian.hpp"
#include "../likelihood/multivariate_gaussian.hpp"
#include "../utils/image_generator.hpp"
#include "../utils/utils.hpp"
#include "hungarian.h"

#ifdef WITH_NMS
    #include "../libs/nms/nms.hpp"
#endif

#include <time.h>
#include <float.h>
#include <vector>
#include <iostream>
#include <random>
#include <chrono>
#include <limits>
#include <algorithm>

using namespace cv;
using namespace std;
using namespace Eigen;


class PHDGaussianMixture {
public:
    int n_particles;
   ~PHDGaussianMixture();
    PHDGaussianMixture(bool verbose = false);
    PHDGaussianMixture();
    void initialize(Mat& current_frame, vector<Rect> detections);
    void update(Mat& image, vector<Rect> detections);
    void predict();
    bool is_initialized();
    
protected:
    mt19937 generator;
    vector<VectorXd> theta_x;
    bool initialized;
    normal_distribution<double> position_random_walk, velocity_random_walk, scale_random_walk;
    Size img_size;
    vector<Target> tracks;
    vector<Target> birth_model;
    RNG rng;
    vector<int> labels, current_labels;
    bool verbose;
};

#endif