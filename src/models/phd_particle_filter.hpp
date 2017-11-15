#ifndef PHD_PARTICLE_FILTER
#define PHD_PARTICLE_FILTER

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include "opencv2/ml.hpp"

#include "../likelihood/gaussian.hpp"
#include "../likelihood/multivariate_gaussian.hpp"
#include "../utils/image_generator.hpp"
#include "../utils/utils.hpp"
#include "../em.hpp"
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

typedef struct particle {
    float x; /** current x coordinate */
    float y; /** current y coordinate */
    float width; /** current width coordinate */
    float height; /** current height coordinate */
    float scale; /** current velocity bounding box scale */
    float x_p; /** current x coordinate */
    float y_p; /** current y coordinate */
    float width_p; /** current width coordinate */
    float height_p; /** current height coordinate */
    float scale_p; /** current velocity bounding box scale */
} particle;

class PHDParticleFilter {
public:
    int n_particles;
    vector<particle> states;
    vector<double> weights;
   ~PHDParticleFilter();
    PHDParticleFilter(int _n_particles, bool verbose = false);
    PHDParticleFilter();
    void initialize(Mat& current_frame, vector<Rect> preDetections);
    void update(Mat& image, vector<Rect> preDetections);
    vector<Target> estimate(Mat& image, bool draw = false);
    void predict();
    void resample();
    void draw_particles(Mat& image, Scalar color);
    bool is_initialized();
    //void auxiliary(Mat& image, vector<Rect> preDetections);
    
protected:
    mt19937 generator;
    vector<VectorXd> theta_x;
    vector<VectorXd> theta_y;
    bool initialized;
    normal_distribution<double> position_random_walk, velocity_random_walk, scale_random_walk;
    Size img_size;
    int max_height, max_width, min_height, min_width;
    int max_x, max_y, min_x, min_y;
    int particles_batch;
    vector<Target> tracks;
    vector<Rect> birth_model;
    RNG rng;
    vector<int> labels, current_labels;
    bool verbose;
};

#endif