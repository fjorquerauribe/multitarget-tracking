#ifndef PARTICLE_FILTER
#define PARTICLE_FILTER


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include "../features/haar.hpp"
#include "../likelihood/gaussian.hpp"
#include "../likelihood/multivariate_gaussian.hpp"
#include "../utils/image_generator.hpp"
#include "../em.hpp"
#include <time.h>
#include <float.h>
#include <vector>
#include <iostream>
#include <random>
#include <chrono>
#include <limits>

extern const float POS_STD;
extern const float VEL_STD;
extern const float SCALE_STD;
extern const float  DT;
extern const float  THRESHOLD;

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

class particle_filter {
public:
    int n_particles;
    vector<particle> states;
    vector<double> weights;
   ~particle_filter();
    particle_filter(int _n_particles);
    particle_filter();
    void initialize(Mat& current_frame, vector<Rect> detections);
    void update(Mat& image, vector<Rect> detections);
    void auxiliary(Mat& image, vector<Rect> detections);
    vector<Rect> estimate(Mat& image,bool draw);
    void predict();
    void resample();
    void draw_particles(Mat& image, Scalar color);
    bool is_initialized();
    
protected:
    mt19937 generator;
    vector<VectorXd> theta_x;
    vector<VectorXd> theta_y;
    bool initialized;
    normal_distribution<double> position_random_walk,velocity_random_walk,scale_random_walk;
    Size im_size;
    int max_height,max_width,min_height,min_width;
    int max_x,max_y,min_x,min_y;
    int particles_batch;
    vector<Target> tracks;
};

#endif