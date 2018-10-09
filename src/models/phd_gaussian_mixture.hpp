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
#include "nms.hpp"
#include "dpp.hpp"

#include <time.h>
#include <float.h>
#include <vector>
#include <set>
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
    PHDGaussianMixture(bool verbose);
    PHDGaussianMixture(bool verbose, double epsilon);
    PHDGaussianMixture(bool verbose, double threshold, int neighbors, double min_scores_sum);
    PHDGaussianMixture();
    void initialize(Mat& current_frame, vector<Rect> detections,VectorXd detectionsWeights);
    void update(Mat& image, vector<Rect> detections, VectorXd detectionsWeights);
    void predict();
    bool is_initialized();
    vector<MyTarget> estimate(Mat& image, bool draw = false);
    
protected:
    mt19937 generator;
    vector<VectorXd> theta_x;
    bool initialized;
    normal_distribution<double> position_random_walk, velocity_random_walk, scale_random_walk;
    Size img_size;
    vector<MyTarget> tracks;
    vector<MyTarget> birth_model;
    RNG rng;
    set<int> labels;
    bool verbose;

    string pruning_method;
    /* DPP parameters*/
    double epsilon;

    /* NMS parameters */
    double threshold, min_scores_sum;
    int neighbors;
};

#endif