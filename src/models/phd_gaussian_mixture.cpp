/**
 * @file phd_gaussian_mixture.cpp
 * @brief phd gaussian mixture
 * @author Sergio Hernandez
 */
#include "phd_gaussian_mixture.hpp"
    
#ifndef PARAMS
    const float POS_STD = 3.0;
    const float SCALE_STD = 3.0;
    const float THRESHOLD = 1000;
    const float SURVIVAL_RATE = 0.9;
    const float CLUTTER_RATE = 2.0;
    const float BIRTH_RATE = 1.0;
    const float DETECTION_RATE = 0.9;
    const float POSITION_LIKELIHOOD_STD = 30.0;
#endif 

PHDGaussianMixture::PHDGaussianMixture() {
}

PHDGaussianMixture::~PHDGaussianMixture() {
    this->states.clear();
    this->weights.clear();
}

bool PHDGaussianMixture::is_initialized() {
    return this->initialized;
}

PHDGaussianMixture::PHDGaussianMixture(bool verbose) {
    this->tracks.clear();
    this->birth_model.clear();
    this->labels.clear();
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    this->generator.seed(seed1);
    this->theta_x.clear();
    RowVectorXd theta_x_pos(2);
    theta_x_pos << POS_STD, POS_STD;
    this->theta_x.push_back(theta_x_pos);
    RowVectorXd theta_x_scale(2);
    theta_x_scale << SCALE_STD, SCALE_STD;
    this->theta_x.push_back(theta_x_scale);
    this->rng(12345);
    this->verbose = verbose;
    this->initialized = false;
}

void PHDGaussianMixture::initialize(Mat& current_frame, vector<Rect> detections) {
    if(detections.size() > 0){
        this->img_size = current_frame.size();
        this->tracks.clear();
        this->birth_model.clear();
        this->labels.clear();

        for (size_t i = 0; i < detections.size(); ++i)
        {
                Target target;
                target.label = i;
                target.color = Scalar(this->rng.uniform(0,255), this->rng.uniform(0,255), this->rng.uniform(0,255));
                target.bbox = detections.at(i);
                target.dx=0.0f;
                target.dy=0.0f;
                this->tracks.push_back(target);
                this->current_labels.push_back(i);
                this->labels.push_back(i);
        }
        this->initialized = true;
    }
}

void PHDGaussianMixture::predict(){
    normal_distribution<double> position_random_x(0.0,theta_x.at(0)(0));
    normal_distribution<double> position_random_y(0.0,theta_x.at(0)(1));
    normal_distribution<double> scale_random_width(0.0,theta_x.at(1)(0));
    normal_distribution<double> scale_random_height(0.0,theta_x.at(1)(1));
    uniform_real_distribution<double> unif(0.0,1.0);

    
    if(this->initialized == true){
        vector<Target> tmp_new_tracks;
        
        for (size_t i = 0; i < this->tracks.size(); i++){
            Target state = this->tracks[i];
            float _x, _y, _width, _height;
            float _dx = position_random_x(this->generator);
            float _dy = position_random_y(this->generator);
            float _dw = scale_random_width(this->generator);
            float _dh = scale_random_height(this->generator);
            _x = MIN(MAX(cvRound(state.bbox.x + _dx), 0), this->img_size.width);
            _y = MIN(MAX(cvRound(state.bbox.y + _dy), 0), this->img_size.height);
            _width = MIN(MAX(cvRound(state.bbox.width + _dw), 0), this->img_size.width);
            _height = MIN(MAX(cvRound(state.bbox.height + _dh), 0), this->img_size.height);
            
            if((_x + _width) < this->img_size.width 
                && _x > 0 
                && (_y + _height) < this->img_size.height 
                && _y > 0
                && _width < this->img_size.width 
                && _height < this->img_size.height 
                && _width > 0 
                && _height > 0 
                && unif(this->generator) < SURVIVAL_RATE){
                state.bbox.x = _x;
                state.bbox.y = _y;
                state.bbox.width = _width;
                state.bbox.height = _height;
                tmp_new_tracks.push_back(state);
            }
        }
        for (size_t j = 0; j < this->birth_model.size(); j++){
            Target state=this->birth_model[j];
            tmp_new_tracks.push_back(state);
        }
        this->tracks.swap(tmp_new_states);
        tmp_new_states.clear();
        tmp_weights.clear();
    }

}


void PHDGaussianMixture::update(Mat& image, vector<Rect> detections)
{
    uniform_real_distribution<double> unif(0.0,1.0);
    double birth_prob = exp(detections.size() * log(BIRTH_RATE) - lgamma(detections.size() + 1.0) - BIRTH_RATE);
    double clutter_prob = exp(detections.size() * log(CLUTTER_RATE) - lgamma(detections.size() + 1.0) - CLUTTER_RATE);
    if(this->initialized && detections.size() > 0){
        vector<Target> new_tracks;
        this->current_labels.clear();
        this->birth_model.clear();
        vector<double> tmp_weights;
        double clutter_prob = (double)CLUTTER_RATE/this->img_size.area();
        int label = 0;
        for (size_t j = 0; j < detections.size(); j++){
            vector<double> IoU;
            Rect current_observation=Rect( detections[j].x, detections[j].y, detections[j].width, detections[j].height);
            Target target;
            target.color = Scalar(this->rng.uniform(0,255), this->rng.uniform(0,255), this->rng.uniform(0,255));
            target.bbox = current_observation
            while( (find(this->labels.begin(), this->labels.end(), label) != this->labels.end()) ||
             (find(this->current_labels.begin(), this->current_labels.end(), label) != this->current_labels.end()) ) label++;
            target.label = label;
            label++;
            for (size_t i = 0; i < this->tracks.size(); ++i){
                Rect current_state=this->tracks[i].bbox;
                double Intersection = (double)(current_state & current_observation).area();
		        double Union=(double)current_state.area()+(double)current_observation.area()-Intersection;
		        IoU.push_back(Intersection/Union);
            }
            if(this->tracks.size()>0){
                double max_iou = *max_element(IoU.begin(), IoU.end());
                double uni_rand = (max_iou > 0.5) ? unif(this->generator) : 1.0;
                if(uni_rand > birth_prob) this->birth_model.push_back(target);
                else{
                    this->current_labels.push_back(label);
                    new_tracks.push_back(target);
                }    
            }
            else{
                this->birth_model.push_back(detections[j]);    
            }
        }
        hungarian_problem_t p;
        /*int **m = Utils::compute_cost_matrix(this->tracks, new_tracks);
        hungarian_init(&p, m, this->tracks.size(), new_tracks.size(), HUNGARIAN_MODE_MINIMIZE_COST);*/
        int **m = Utils::compute_overlap_matrix(this->tracks, new_tracks);
        hungarian_init(&p, m, this->tracks.size(), new_tracks.size(), HUNGARIAN_MODE_MAXIMIZE_UTIL);
        
        hungarian_solve(&p);
        for (size_t i = 0; i < this->tracks.size(); ++i)
        {
            for (size_t j = 0; j < new_tracks.size(); ++j)
            {
                if (p.assignment[i][j] == HUNGARIAN_ASSIGNED)
                {
                    new_tracks.at(j).label = this->tracks.at(i).label;
                    new_tracks.at(j).color = this->tracks.at(i).color;
                    break;
                }
            }
        }
        if(this->verbose){
            cout << "New Detections: "<< detections.size() << endl;
            cout << "New Born Targets: "<< this->birth_model.size() << endl;
        }
        this->states.swap(tmp_new_states);
    }
}