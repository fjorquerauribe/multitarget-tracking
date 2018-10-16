/**
 * @file phd_gaussian_mixture.cpp
 * @brief phd gaussian mixture
 * @author Sergio Hernandez
 */
#include "phd_gaussian_mixture.hpp"
    
#ifndef PARAMS
    const float POS_STD = 3.0;
    const float SCALE_STD = 3.0;
    const float THRESHOLD = 10;
    const float SURVIVAL_RATE = 1.0;
    const float SURVIVAL_DECAY = 1.0;
    const float CLUTTER_RATE = 2.0;
    const float BIRTH_RATE = 0.1;
    const float DETECTION_RATE = 0.5;
    const float POSITION_LIKELIHOOD_STD = 30.0;
#endif 

PHDGaussianMixture::PHDGaussianMixture() {
}

PHDGaussianMixture::~PHDGaussianMixture(){
    this->tracks.clear();
    this->birth_model.clear();
    this->labels.clear();
}

bool PHDGaussianMixture::is_initialized() {
    return this->initialized;
}

PHDGaussianMixture::PHDGaussianMixture(bool verbose, double epsilon): PHDGaussianMixture(verbose) {
    this->pruning_method = "nms";
    this->epsilon = epsilon;
}

PHDGaussianMixture::PHDGaussianMixture(bool verbose, double threshold, 
    int neighbors, double min_scores_sum): PHDGaussianMixture(verbose){
    this->pruning_method = "nms";
    this->threshold = threshold;
    this->neighbors = neighbors;
    this->min_scores_sum = min_scores_sum;
}

PHDGaussianMixture::PHDGaussianMixture(bool verbose){
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

void PHDGaussianMixture::initialize(Mat& current_frame, vector<Rect> detections, VectorXd detectionsWeights) {
    if(detections.size() > 0){
        this->img_size = current_frame.size();
        this->tracks.clear();
        this->birth_model.clear();
        this->labels.clear();

        for (size_t i = 0; i < detections.size(); ++i)
        {
                MyTarget target;
                target.label = i;
                target.color = Scalar(this->rng.uniform(0,255), this->rng.uniform(0,255), this->rng.uniform(0,255));
                target.bbox = detections.at(i);
                target.survival_rate = SURVIVAL_RATE;
                target.score = detectionsWeights(i);
                this->tracks.push_back(target);
                this->labels.insert(i);
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
        vector<MyTarget> tmp_new_tracks;
        
        for (size_t i = 0; i < this->tracks.size(); i++){
            MyTarget track = this->tracks[i];
            float _x, _y, _width, _height;
            float _dx = position_random_x(this->generator);
            float _dy = position_random_y(this->generator);
            float _dw = scale_random_width(this->generator);
            float _dh = scale_random_height(this->generator);
            _x = MIN(MAX(cvRound(track.bbox.x), 0), this->img_size.width);
            _y = MIN(MAX(cvRound(track.bbox.y  ), 0), this->img_size.height);
            _width = MIN(MAX(cvRound(track.bbox.width), 0), this->img_size.width);
            _height = MIN(MAX(cvRound(track.bbox.height ), 0), this->img_size.height);
            
            if((_x + _width) < this->img_size.width 
                && _x > 0 
                && (_y + _height) < this->img_size.height 
                && _y > 0
                && _width < this->img_size.width 
                && _height < this->img_size.height 
                && _width > 0 
                && _height > 0 
                && unif(this->generator) < track.survival_rate){
                track.bbox.x = _x;
                track.bbox.y = _y;
                track.bbox.width = _width;
                track.bbox.height = _height;
                tmp_new_tracks.push_back(track);
            }
        }
        for (size_t j = 0; j < this->birth_model.size(); j++){
            MyTarget track = this->birth_model[j];
            tmp_new_tracks.push_back(track);
        }
        this->birth_model.clear();
        this->tracks.swap(tmp_new_tracks);
        tmp_new_tracks.clear();
    }

}


void PHDGaussianMixture::update(Mat& image, vector<Rect> detections,  VectorXd detectionsWeights)
{
    uniform_real_distribution<double> unif(0.0,1.0);
    this->birth_model.clear();
    /*double birth_prob = exp(detections.size() * log(BIRTH_RATE) - lgamma(detections.size() + 1.0) - BIRTH_RATE);
    double clutter_prob = exp(detections.size() * log(CLUTTER_RATE) - lgamma(detections.size() + 1.0) - CLUTTER_RATE);
    double birth_prob = BIRTH_RATE/detections.size();*/
    if(this->initialized && detections.size() > 0){
        vector<MyTarget> new_detections;
        vector<MyTarget> new_tracks;
        set<int> new_labels;
        vector<double> tmp_weights;
        int label = 0;
        for (size_t j = 0; j < detections.size(); j++){
            MyTarget target;
            target.color = Scalar(this->rng.uniform(0,255), this->rng.uniform(0,255), this->rng.uniform(0,255));
            target.bbox = detections[j];
            target.survival_rate = SURVIVAL_RATE;
            target.score = detectionsWeights(j);
            /*while( this->labels.find(label)!= this->labels.end() ) label++;
            target.label = label;
            this->labels.insert(label);*/
            new_detections.push_back(target);
        }
        hungarian_problem_t p;
        double diagonal = sqrt( pow(image.rows,2) + pow(image.cols,2));
        double area = image.rows * image.cols;
        int **m = Utils::compute_cost_matrix(this->tracks, new_detections, diagonal, area);
        hungarian_init(&p, m, this->tracks.size(), new_detections.size(), HUNGARIAN_MODE_MINIMIZE_COST);
        /*int **m = Utils::compute_overlap_matrix(this->tracks, new_detections);
        hungarian_init(&p, m, this->tracks.size(), new_tracks.size(), HUNGARIAN_MODE_MAXIMIZE_UTIL);*/
        hungarian_solve(&p);    
        for (size_t i = 0; i < this->tracks.size(); ++i)
        {
            int no_assignment_count = 0;
            for (size_t j = 0; j < new_detections.size(); ++j)
            {
                if(this->verbose){
                    cout << "track  : " << this->tracks.at(i).label 
                         << ", new detections  : " << new_detections.at(j).label
                         << ", cost  : " << m[i][j]
                         << ", assignment : " << p.assignment[i][j] << endl;
                }
                if (p.assignment[i][j] == HUNGARIAN_ASSIGNED && m[i][j] < THRESHOLD)
                {
                    new_detections.at(j).label = this->tracks.at(i).label;
                    new_detections.at(j).color = this->tracks.at(i).color;
                    //new_detections.at(j).feature.noalias()=0.1*new_detections.at(j).feature.array()+0.9*this->tracks.at(i).feature.array();
                    new_tracks.push_back(new_detections.at(j));
                    break;
                }
                no_assignment_count++;
                //this->labels.insert(new_detections.at(j).label);
            }
            if(no_assignment_count == (int)new_detections.size()) {
                this->tracks.at(i).survival_rate = exp(SURVIVAL_DECAY * (-1.0 + this->tracks.at(i).survival_rate * 0.9));
                new_tracks.push_back(this->tracks.at(i));
            }  
        }
        for (size_t j = 0; j < new_detections.size(); ++j)
        {
            int no_assignment_count = 0;
            for (size_t i = 0; i < this->tracks.size(); ++i)
            {
                if (p.assignment[i][j] == HUNGARIAN_ASSIGNED)
                {
                    break;
                }
                no_assignment_count++;
            }
            if(no_assignment_count == (int)this->tracks.size() && unif(this->generator) > BIRTH_RATE) {
                while( this->labels.find(label) != this->labels.end() ) label++;
                new_detections.at(j).label = label;
                this->labels.insert(label);
                birth_model.push_back(new_detections.at(j));
            }  
        }
        if(this->verbose){
            cout << "Current Targets: "<< this->tracks.size() << endl;
            cout << "New Detections: "<< detections.size() << endl;
            cout << "New Born Targets: "<< this->birth_model.size() << endl;
            cout << "Updated Targets: "<< new_tracks.size() << endl;
        }

        if(this->pruning_method.compare("nms") == 0){
            nms4(new_tracks, this->tracks, this->threshold, this->neighbors, this->min_scores_sum);
        }
        else if(this->pruning_method.compare("dpp") == 0){
            DPP dpp = DPP();
            this->tracks = dpp.run(new_tracks, this->epsilon, this->img_size); // epsilon: 0.5 | q: score
        }
        else {
            this->tracks.swap(new_tracks);
        }
        new_tracks.clear();
        new_labels.clear();
    }
}

vector <MyTarget> PHDGaussianMixture::estimate(Mat& image, bool draw)
{
    if (draw){
        for (size_t i = 0; i < this->tracks.size(); ++i){
            Rect current_estimate = this->tracks.at(i).bbox;
            rectangle( image, current_estimate, this->tracks.at(i).color, 3, LINE_8  );
            rectangle( image, Point(current_estimate.x, current_estimate.y - 10),
                Point(current_estimate.x + current_estimate.width, current_estimate.y + 20),
                this->tracks.at(i).color, -1, 8, 0 );
            //string disp=to_string(this->tracks.at(i).label)+","+to_string( this->tracks.at(i).survival_rate).substr(0,3);
            putText(image,to_string( this->tracks.at(i).label), Point(current_estimate.x + 5,current_estimate.y + 12), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1 );
            //putText(image,disp,Point(current_estimate.x+5,current_estimate.y+12),FONT_HERSHEY_SIMPLEX,0.3,Scalar(255,255,255),1);
        }
    }
    return this->tracks;
}