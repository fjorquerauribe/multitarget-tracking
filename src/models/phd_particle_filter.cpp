/**
 * @file phd_particle_filter.cpp
 * @brief phd particle filter
 * @author Sergio Hernandez
 */
#include "phd_particle_filter.hpp"

#ifndef PARAMS
const float POS_STD = 3.0;
const float SCALE_STD = 3.0;
const float THRESHOLD = 1000;
const float SURVIVAL_RATE = 0.9;
const float CLUTTER_RATE = 1.0e-3;
const float BIRTH_RATE = 5e-6;
const float DETECTION_RATE = 0.7;
const float POSITION_LIKELIHOOD_STD = 30.0;
#endif 

PHDParticleFilter::PHDParticleFilter() {
}

PHDParticleFilter::~PHDParticleFilter() {
    this->states.clear();
    this->weights.clear();
}

bool PHDParticleFilter::is_initialized() {
    return this->initialized;
}

PHDParticleFilter::PHDParticleFilter(int _n_particles) {
    this->states.clear();
    this->weights.clear();
    this->birth_model.clear();
    this->n_particles = _n_particles;
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    this->generator.seed(seed1);
    this->theta_x.clear();
    RowVectorXd theta_x_pos(2);
    theta_x_pos << POS_STD,POS_STD;
    this->theta_x.push_back(theta_x_pos);
    RowVectorXd theta_x_scale(2);
    theta_x_scale << SCALE_STD,SCALE_STD;
    this->theta_x.push_back(theta_x_scale);
    this->max_height = 100;
    this->max_width = 40;
    this->min_height = 1e6;
    this->min_width = 1e6;
    this->max_x = 0;
    this->max_y = 0;
    this->min_x = 1e6;
    this->min_y = 1e6;
    this->initialized = false;
}

void PHDParticleFilter::initialize(Mat& current_frame, vector<Rect> detections) {
    this->img_size = current_frame.size();
    normal_distribution<double> position_random_x(0.0,theta_x.at(0)(0));
    normal_distribution<double> position_random_y(0.0,theta_x.at(0)(1));
    normal_distribution<double> scale_random_width(0.0,theta_x.at(1)(0));
    normal_distribution<double> scale_random_height(0.0,theta_x.at(1)(1));
    this->states.clear();
    this->weights.clear();
    this->particles_batch = (int)this->n_particles;
    double weight = (double)1.0f/this->n_particles;
    
    for(unsigned int j = 0; j < detections.size(); j++){
        this->max_width = MAX(detections[j].width, this->max_width);
        this->max_height = MAX(detections[j].height, this->max_height);
        this->min_width = MIN(detections[j].width, this->min_width);
        this->min_height = MIN(detections[j].height, this->min_height);
        this->max_x = MAX(detections[j].x, this->max_x);
        this->max_y = MAX(detections[j].y, this->max_y);
        this->min_x = MIN(detections[j].x, this->min_x);
        this->min_y = MIN(detections[j].y, this->min_y);
        for (int i = 0; i < this->particles_batch; i++){
            particle state;
            float _x, _y, _width, _height;
            float _dx = position_random_x(this->generator);
            float _dy =+ position_random_y(this->generator);
            float _dw = 0.0f;//scale_random_width(generator);
            float _dh = 0.0f;//scale_random_height(generator);
            _x = MIN(MAX(cvRound(detections[j].x + _dx), 0), this->img_size.width);
            _y = MIN(MAX(cvRound(detections[j].y + _dy), 0), this->img_size.height);
            _width = MIN(MAX(cvRound(detections[j].width + _dw), 0), this->img_size.width);
            _height = MIN(MAX(cvRound(detections[j].height + _dh), 0), this->img_size.height);
            state.x = _x;
            state.y = _y;
            state.width = _width;
            state.height = _height;
            state.scale = 1.0;
            this->states.push_back(state);
            this->weights.push_back(weight);
        }
    }
    
    uniform_int_distribution<int> random_x(this->min_x, this->max_x);
    uniform_int_distribution<int> random_y(this->min_y, this->max_y);
    uniform_int_distribution<int> random_w(this->min_width, this->max_width);
    uniform_int_distribution<int> random_h(this->min_height, this->max_height);
    /*for (int i = 0; i < remaining_batch; i++){
        particle state;
        state.width = random_w(this->generator);
        state.height = random_h(this->generator);
        state.x = cvRound(random_x(this->generator));
        state.y = cvRound(random_y(this->generator));
        state.scale = 1.0;
        this->states.push_back(state);
        this->weights.push_back(1/100.f);
    } */   
    this->initialized = true;
    Scalar phd_estimate = sum(this->weights);
    cout << "initial estimated number : "<< cvRound(phd_estimate[0]) << endl; 
}

void PHDParticleFilter::predict(){
    normal_distribution<double> position_random_x(0.0,theta_x.at(0)(0));
    normal_distribution<double> position_random_y(0.0,theta_x.at(0)(1));
    normal_distribution<double> scale_random_width(0.0,theta_x.at(1)(0));
    normal_distribution<double> scale_random_height(0.0,theta_x.at(1)(1));
    uniform_real_distribution<double> unif(0.0,1.0);
    uniform_int_distribution<int> random_x(this->min_x, this->max_x);
    uniform_int_distribution<int> random_y(this->min_y, this->max_y);
    uniform_int_distribution<int> random_w(this->min_width, this->max_width);
    uniform_int_distribution<int> random_h(this->min_height, this->max_height);
    if(this->initialized == true){
        vector<particle> tmp_new_states;
        vector<double> tmp_weights;
        for (unsigned int i = 0; i < this->states.size(); i++){
            particle state = this->states[i];
            float _x, _y, _width, _height;
            float _dx = position_random_x(this->generator);
            float _dy = position_random_y(this->generator);
            float _dw = scale_random_width(this->generator);
            float _dh = scale_random_height(this->generator);
            _x = MIN(MAX(cvRound(state.x + _dx), 0), this->img_size.width);
            _y = MIN(MAX(cvRound(state.y + _dy), 0), this->img_size.height);
            _width = MIN(MAX(cvRound(state.width + _dw), 0), this->img_size.width);
            _height = MIN(MAX(cvRound(state.height + _dh), 0), this->img_size.height);
            
            if((_x + _width) < this->img_size.width 
                && _x > 0 
                && (_y + _height) < this->img_size.height 
                && _y > 0
                && _width < this->img_size.width 
                && _height < this->img_size.height 
                && _width > 0 
                && _height > 0 
                && unif(this->generator) < SURVIVAL_RATE){
                //&& unif(generator)<SURVIVAL_RATE
                state.x_p = state.x;
                state.y_p = state.y;
                state.width_p = state.width;
                state.height_p = state.height;       
                state.x = _x;
                state.y = _y;
                state.width = _width;
                state.height = _height;
                state.scale_p = state.scale;
                state.scale = 2 * state.scale - state.scale_p + scale_random_width(this->generator);
                Rect box(state.x, state.y, state.width, state.height);
                tmp_new_states.push_back(state);
                tmp_weights.push_back(SURVIVAL_RATE*this->weights.at(i));
            }
        }
        double lambda_birth = this->img_size.area() * BIRTH_RATE;
        //poisson_distribution<int> birth_num(lambda_birth);
        //int J_k = birth_num(this->generator);    
        if(this->birth_model.size()>0){
            for (unsigned int j = 0; j < this->birth_model.size(); j++){
                for (int k = 0; k < this->particles_batch; k++){
                    particle state;
                    state.width = this->birth_model[j].width+ scale_random_width(this->generator);
                    state.height = this->birth_model[j].height+ scale_random_height(this->generator);
                    state.x = this->birth_model[j].x+ position_random_x(this->generator);;
                    state.y = this->birth_model[j].y+ position_random_y(this->generator);;
                    Rect box(state.x, state.y, state.width, state.height);
                    tmp_new_states.push_back(state);
                    tmp_weights.push_back((double)lambda_birth/(this->birth_model.size()*this->particles_batch));
                }
            }   
        }
        /*else{
            double lambda_birth = this->img_size.area() * BIRTH_RATE;
            poisson_distribution<int> birth_num(lambda_birth);
            int J_k = birth_num(this->generator);
            for (int j = 0; j < J_k; j++){
               for (int k = 0; k < particles_batch; k++){
                    particle state;
                    state.width = cvRound(random_w(this->generator));
                    state.height = cvRound(random_h(this->generator));
                    state.x = cvRound(random_x(this->generator));
                    state.y = cvRound(random_y(this->generator));
                    Rect box(state.x, state.y, state.width, state.height);
                    tmp_new_states.push_back(state);
                    tmp_weights.push_back((double)1.0f/J_k);
                }
            }
        }*/
        this->states.swap(tmp_new_states);
        this->weights.swap(tmp_weights);
        Scalar phd_estimate = sum(this->weights);
        tmp_new_states.clear();// = vector<particle>();
        tmp_weights.clear();// = vector<double>();
        cout << "predicted target number : "<< cvRound(phd_estimate[0]) << endl; 
        //cout << "predicted birth number : "<< J_k << endl; 
    }
}

void PHDParticleFilter::draw_particles(Mat& image, Scalar color = Scalar(0,255,255)){
    for (unsigned int i = 0; i < this->states.size(); i++){
        particle state = this->states[i];
        Point pt1, pt2;
        pt1.x = cvRound(state.x);
        pt1.y = cvRound(state.y);
        pt2.x = cvRound(state.x + state.width);
        pt2.y = cvRound(state.y + state.height);
        rectangle( image, pt1, pt2, color, 2, LINE_AA );
    }
}

void PHDParticleFilter::update(Mat& image, vector<Rect> detections)
{
    //cout << "states size: " << this->states.size() << endl;
    if(detections.size() > 0){
        //this->birth_model.clear();
        vector<double> tmp_weights;
        MatrixXd cov = POSITION_LIKELIHOOD_STD * POSITION_LIKELIHOOD_STD * MatrixXd::Identity(4, 4);
        //cout << "detections : " << detections.size() << endl;
        MatrixXd observations = MatrixXd::Zero(detections.size(),4);
        
        double clutter_prob = (double)CLUTTER_RATE/this->img_size.area();
        
        for (unsigned int j = 0; j < detections.size(); j++){
            this->max_width = MAX(detections[j].width, this->max_width);
            this->max_height = MAX(detections[j].height, this->max_height);
            this->min_width = MIN(detections[j].width, this->min_width);
            this->min_height = MIN(detections[j].height, this->min_height);
            this->max_x = MAX(detections[j].x, this->max_x);
            this->max_y = MAX(detections[j].y, this->max_y);
            this->min_x = MIN(detections[j].x, this->min_x);
            this->min_y = MIN(detections[j].y, this->min_y);
            observations.row(j) << detections[j].x, detections[j].y, detections[j].width, detections[j].height;
            this->birth_model.push_back(detections[j]);
        }

        MatrixXd psi(this->states.size(), detections.size());
        for (unsigned int i = 0; i < states.size(); ++i)
        {
            particle state = this->states[i];
            VectorXd mean(4);
            mean << state.x, state.y, state.width, state.height;
            MatrixXd cov = POSITION_LIKELIHOOD_STD * POSITION_LIKELIHOOD_STD * MatrixXd::Identity(4, 4);
            MVNGaussian gaussian(mean, cov);
            double weight = this->weights[i];
            psi.row(i) = DETECTION_RATE * weight * gaussian.log_likelihood(observations).array().exp();
        }
        VectorXd tau = VectorXd::Zero(detections.size());
        tau = clutter_prob+psi.colwise().sum().array();
        for (size_t i = 0; i < this->weights.size(); ++i)
        {
            double weight = this->weights[i];
            tmp_weights.push_back((1 - DETECTION_RATE)*weight + psi.row(i).cwiseQuotient(tau.transpose()).sum() );
        }
        this->weights.swap(tmp_weights);
        resample();
        Scalar phd_estimate = sum(this->weights);
        cout << "Updated target number : "<< cvRound(phd_estimate[0]) << endl;  
        tmp_weights.clear();
    }
}

void PHDParticleFilter::resample(){
    cout << "resample!" << endl;
    int L_k = this->states.size();
    vector<double> cumulative_sum(L_k);
    vector<double> normalized_weights(L_k);
    vector<double> log_weights(L_k);
    vector<double> squared_normalized_weights(L_k);
    uniform_real_distribution<double> unif_rnd(0.0,1.0); 
    for (int i = 0; i < L_k; i++) {
        log_weights[i]= log(this->weights[i]);
    }
    double logsumexp = 0.0;
    double max_value = *max_element(log_weights.begin(), log_weights.end());
    for (int i = 0; i < L_k; i++) {
        normalized_weights[i] = exp(log_weights[i] - max_value);
        logsumexp += normalized_weights[i];
    }
    for (int i = 0; i < L_k; i++) {
        normalized_weights.at(i) = normalized_weights.at(i)/logsumexp;
    }
    
    for (int i = 0; i < L_k; i++) {
        squared_normalized_weights.at(i) = normalized_weights.at(i) * normalized_weights.at(i);
        if (i == 0) {
            cumulative_sum.at(i) = normalized_weights.at(i);
        } else {
            cumulative_sum.at(i) = cumulative_sum.at(i - 1) + normalized_weights.at(i);
        }
    }
    Scalar sum_squared_weights = sum(squared_normalized_weights);
    Scalar phd_estimate = sum(this->weights);
    int N_k=min(cvRound(this->particles_batch*phd_estimate[0]),500);
    double ESS=(1.0f/sum_squared_weights[0]);
    if(isless(ESS,(float)THRESHOLD)){
        vector<particle> tmp_new_states;
        vector<double> tmp_weights;
        for (int i=0; i<N_k; i++) {
            double uni_rand = unif_rnd(this->generator);
            vector<double>::iterator pos = lower_bound(cumulative_sum.begin(), cumulative_sum.end(), uni_rand);
            int ipos = distance(cumulative_sum.begin(), pos);
            particle state = this->states[ipos];
            tmp_new_states.push_back(state);
            tmp_weights.push_back((double)1.0/this->particles_batch);
        }
        this->states.swap(tmp_new_states);
        this->weights.swap(tmp_weights);
        tmp_new_states = vector<particle>();
        tmp_weights = vector<double>();
    }
    cumulative_sum.clear();
    squared_normalized_weights.clear();
}

vector<Rect> PHDParticleFilter::estimate(Mat& image, bool draw = false){
    vector<Target> new_tracks;
    Scalar phd_estimate = sum(this->weights);
    int num_targets = cvRound(phd_estimate[0]);
    vector<Rect> estimates;
    if(num_targets>0)
    {
        MatrixXd data((int)this->states.size(),4);
        for (unsigned int j = 0; j < this->states.size(); j++){
            data.row(j) << this->states[j].x, this->states[j].y, this->states[j].width, this->states[j].height;
        }
        EM mixture(data, num_targets);
        MatrixXd cost_matrix = MatrixXd::Zero(num_targets, this->tracks.size());
        mixture.fit(10);
        vector<VectorXd> eigen_estimates = mixture.getMeans();
        for(unsigned int k = 0; k < eigen_estimates.size(); k++){
            VectorXd vec = eigen_estimates[k];
            for(unsigned int n = 0; n < this->tracks.size(); n++){
                VectorXd eigen_track;
                eigen_track << this->tracks[n].bbox.x, this->tracks[n].bbox.y, this->tracks[n].bbox.width, this->tracks[n].bbox.height;
            }
            Point pt1,pt2;
            pt1.x=cvRound(vec(0));
            pt1.y=cvRound(vec(1));
            float _width=cvRound(vec(2));
            float _height=cvRound(vec(3));
            pt2.x=cvRound(pt1.x+_width);
            pt2.y=cvRound(pt1.y+_height); 
            if(vec[0]<this->img_size.width && vec[0]>=0 && vec[1]<this->img_size.height && vec[1]>=0){
                   if(draw) rectangle( image, pt1,pt2, Scalar(0,0,255), 2, LINE_AA );
                Rect estimate=Rect(pt1.x,pt1.y,_width,_height);
                estimates.push_back(estimate);
                //Target target;
                //target.bbox=estimate;
                //target.label=k;
                //this->tracks.push_back(target);
            }
        }
    }
    
    return estimates;
}

/*void PHDParticleFilter::auxiliary(Mat& image, vector<Rect> detections)
{
    vector<double> aux_weights,tmp_weights;
    MatrixXd cov = POSITION_LIKELIHOOD_STD * POSITION_LIKELIHOOD_STD * MatrixXd::Identity(4, 4);
    MatrixXd observations = MatrixXd::Zero(detections.size(), 4);
    for (unsigned int j = 0; j < detections.size(); j++){
        this->max_width = MAX(detections[j].width, this->max_width);
        this->max_height = MAX(detections[j].height, this->max_height);
        this->min_width = MIN(detections[j].width, this->min_width);
        this->min_height = MIN(detections[j].height, this->min_height);
        this->max_x = MAX(detections[j].x, this->max_x);
        this->max_y = MAX(detections[j].y, this->max_y);
        this->min_x = MIN(detections[j].x, this->min_x);
        this->min_y = MIN(detections[j].y, this->min_y);
        observations.row(j) << detections[j].x, detections[j].y, detections[j].width,detections[j].height;
    }
    for (int i = 0; i < n_particles; i++){
        particle state = this->states[i];
        double weight = this->weights[i];
        VectorXd  mean(4);
        mean << state.x, state.y, state.width,state.height;
        MVNGaussian gaussian(mean,cov);
        VectorXd psi = DETECTION_RATE * weight * gaussian.log_likelihood(observations);
        double sumexp = 0.0;
        double max_value = psi.maxCoeff();
        for (unsigned int j = 0; j < detections.size(); j++) {
            sumexp += exp(psi[j] - max_value);
        }
        double norm_const = max_value + log(sumexp);
        VectorXd phd_update = (psi.array() - norm_const - CLUTTER_RATE/(this->img_size.width * this->img_size.height)).exp().matrix();
        aux_weights.push_back(phd_update.array().sum() + (1 - DETECTION_RATE) * weight);
    }
    //this->weights.swap(tmp_weights);
    //
    resample();
    predict();

    for (int i = 0;i < n_particles; i++){
        particle state = this->states[i];
        double weight = this->weights[i];
        VectorXd  mean(4);
        mean << state.x, state.y, state.width,state.height;
        MVNGaussian gaussian(mean,cov);
        VectorXd psi = DETECTION_RATE * weight * gaussian.log_likelihood(observations);
        double sumexp = 0.0;
        double max_value = psi.maxCoeff();
        for (unsigned int j = 0; j < detections.size(); j++) {
            sumexp += exp(psi[j] - max_value);
        }
        double norm_const = max_value + log(sumexp);
        VectorXd phd_update = (psi.array() - norm_const - CLUTTER_RATE * this->img_size.area()).exp().matrix();
        tmp_weights.push_back(phd_update.array().sum() + (1 - DETECTION_RATE) * weight);
    }
    for (unsigned int k = 0; k < this->weights.size(); k++) {
        this->weights[k] = tmp_weights.at(k)/aux_weights.at(k);
    }
    tmp_weights.clear();
    aux_weights.clear();
    resample();
}*/
