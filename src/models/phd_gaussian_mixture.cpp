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
                tmp_new_states.push_back(state);
            }
        }

        if(this->birth_model.size() > 0){
            //double birth_prob = BIRTH_RATE/(double)birth_model.size();
            for (size_t j = 0; j < this->birth_model.size(); j++){
                for (int k = 0; k < this->particles_batch; k++){
                        particle state;
                        state.width = this->birth_model[j].width + scale_random_width(this->generator);
                        state.height = this->birth_model[j].height + scale_random_height(this->generator);
                        state.x = this->birth_model[j].x + position_random_x(this->generator);
                        state.y = this->birth_model[j].y + position_random_y(this->generator);
                        Rect box(state.x, state.y, state.width, state.height);
                        tmp_new_states.push_back(state);
                        tmp_weights.push_back(1.0f/((double)this->particles_batch));
                }
            }
        }

        this->states.swap(tmp_new_states);
        this->weights.swap(tmp_weights);
        tmp_new_states.clear();
        tmp_weights.clear();
    }

    if(this->verbose){
        Scalar phd_estimate = sum(this->weights);
        cout << "Predicted target number: "<< cvRound(phd_estimate[0]) << endl;
    }
}


void PHDGaussianMixture::update(Mat& image, vector<Rect> detections)
{
    uniform_real_distribution<double> unif(0.0,1.0);
    if(this->initialized && detections.size() > 0){
        this->birth_model.clear();
        vector<double> tmp_weights;
        MatrixXd observations = MatrixXd::Zero(detections.size(), 4);
        double clutter_prob = (double)CLUTTER_RATE/this->img_size.area();
        for (size_t j = 0; j < detections.size(); j++){
            vector<double> IoU;
            Rect current_observation=Rect( detections[j].x, detections[j].y, detections[j].width, detections[j].height);
            for (size_t i = 0; i < this->tracks.size(); ++i){
                Rect current_state=this->tracks[i].bbox;
                double Intersection = (double)(current_state & current_observation).area();
		        double Union=(double)current_state.area()+(double)current_observation.area()-Intersection;
		        IoU.push_back(Intersection/Union);
            }
            if(this->tracks.size()>0){
                double max_iou = *max_element(IoU.begin(), IoU.end());
                double uni_rand = (max_iou > 0.5) ? unif(this->generator) : 1.0;
                if(uni_rand > 0.9) this->birth_model.push_back(detections[j]);    
            }
            else{
                this->birth_model.push_back(detections[j]);    
            }
        }
        if(this->verbose){
            cout << "New Detections: "<< detections.size() << endl;
            cout << "New Born Targets: "<< this->birth_model.size() << endl;
        }
        MatrixXd psi(this->states.size(), detections.size());
        for (size_t i = 0; i < this->states.size(); ++i)
        {
            particle state = this->states[i];
            VectorXd mean(4);
            mean << state.x, state.y, state.width, state.height;
            MatrixXd cov = POSITION_LIKELIHOOD_STD * POSITION_LIKELIHOOD_STD * MatrixXd::Identity(4, 4);
            MVNGaussian gaussian(mean, cov);
            psi.row(i) = DETECTION_RATE * this->weights[i] * gaussian.log_likelihood(observations).array().exp();
        }

        VectorXd tau = VectorXd::Zero(detections.size());
        tau = clutter_prob + psi.colwise().sum().array();
        for (size_t i = 0; i < this->weights.size(); ++i)
        {
            tmp_weights.push_back((1.0f - DETECTION_RATE) * this->weights[i] + psi.row(i).cwiseQuotient(tau.transpose()).sum() );
        }
        
        this->weights.swap(tmp_weights);
        if(this->verbose){
            Scalar phd_estimate = sum(this->weights);
            cout << "Update target number: "<< cvRound(phd_estimate[0]) << endl;
        }
        resample();
        tmp_weights.clear();
    }
}

vector<Target> PHDGaussianMixture::estimate(Mat& image, bool draw){
    if(this->initialized){
        Scalar phd_estimate = sum(this->weights);
        int num_targets = cvRound(phd_estimate[0]);
        vector<Target> new_tracks;
        if(num_targets > 0)
        {
            /********************** EM **********************/
            Mat data, em_labels, emMeans;
            data = Mat::zeros((int)this->states.size(),6, CV_64F);
            for (size_t j = 0; j < this->states.size(); j++){
                data.at<double>(j,0) = this->states[j].x;
                data.at<double>(j,1) = this->states[j].y;
                data.at<double>(j,2) = this->states[j].width;
                data.at<double>(j,3) = this->states[j].height;
                data.at<double>(j,4) = this->states[j].x_p;
                data.at<double>(j,5) = this->states[j].y_p;
            }   
            Ptr<cv::ml::EM> em_model = cv::ml::EM::create();
            em_model->setClustersNumber(num_targets);
            em_model->setCovarianceMatrixType(cv::ml::EM::COV_MAT_DIAGONAL);
            em_model->setTermCriteria(TermCriteria(TermCriteria::COUNT, 10, 0.1));// 10,0.1
            em_model->trainEM(data, noArray(), em_labels, noArray());
            emMeans = em_model->getMeans();
            
            int label = 0;
            for (int i = 0; i < emMeans.rows; ++i)
            {
                Target target;
                target.color = Scalar(this->rng.uniform(0,255), this->rng.uniform(0,255), this->rng.uniform(0,255));
                target.bbox = Rect(emMeans.at<double>(i,0), emMeans.at<double>(i,1), emMeans.at<double>(i,2), emMeans.at<double>(i,3));
                target.dx=emMeans.at<double>(i,0)-emMeans.at<double>(i,4);
                target.dy=emMeans.at<double>(i,1)-emMeans.at<double>(i,5);
                while( (find(this->labels.begin(), this->labels.end(), label) != this->labels.end()) ||
                 (find(this->current_labels.begin(), this->current_labels.end(), label) != this->current_labels.end()) ) label++;
                target.label = label;
                this->current_labels.push_back(label);
                label++;
                new_tracks.push_back(target);
            }
            /*******************************************************/
            
            if (this->tracks.size() > 0)
            {
                hungarian_problem_t p;
                int **m = Utils::compute_cost_matrix(this->tracks, new_tracks);
                hungarian_init(&p, m, this->tracks.size(), new_tracks.size(), HUNGARIAN_MODE_MINIMIZE_COST);
                /*int **m = Utils::compute_overlap_matrix(this->tracks, new_tracks);
                hungarian_init(&p, m, this->tracks.size(), new_tracks.size(), HUNGARIAN_MODE_MAXIMIZE_UTIL);*/
                
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
            }
    
            this->tracks.swap(new_tracks);
            this->current_labels.clear();
            for(size_t i = 0; i < this->tracks.size(); i++) this->current_labels.push_back(this->tracks.at(i).label);
            
            this->labels.insert( this->labels.end(), this->current_labels.begin(), this->current_labels.end() );
            new_tracks.clear();
    
            if (draw)
            {
                for (size_t i = 0; i < this->tracks.size(); ++i)
                {
                    //rectangle(image, this->tracks.at(i).bbox, this->tracks.at(i).color, 3, LINE_8);
                    Rect current_estimate=Rect(this->tracks.at(i).bbox.x,this->tracks.at(i).bbox.y,this->tracks.at(i).bbox.width,this->tracks.at(i).bbox.height);
                    rectangle( image,current_estimate, this->tracks.at(i).color, 3, LINE_8  );
                    rectangle( image,Point(current_estimate.x,current_estimate.y-10),
                            Point(current_estimate.x+current_estimate.width,current_estimate.y+20),
                                this->tracks.at(i).color, -1,8,0 );
                    putText(image,to_string( this->current_labels.at(i)),Point(current_estimate.x+5,current_estimate.y+12),FONT_HERSHEY_SIMPLEX,0.5,Scalar(255,255,255),1);
                }
            }
        }
        else
        {
            this->tracks.clear();
            this->current_labels.clear();
        }    
    }
    return this->tracks;
}
