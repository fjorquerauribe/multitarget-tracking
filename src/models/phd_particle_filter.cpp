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
    const float CLUTTER_RATE = 1.0e-2;
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

PHDParticleFilter::PHDParticleFilter(int _n_particles, bool verbose) {
    this->states.clear();
    this->weights.clear();
    this->birth_model.clear();
    this->labels.clear();
    this->n_particles = _n_particles;
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    this->generator.seed(seed1);
    this->theta_x.clear();
    RowVectorXd theta_x_pos(2);
    theta_x_pos << POS_STD, POS_STD;
    this->theta_x.push_back(theta_x_pos);
    RowVectorXd theta_x_scale(2);
    theta_x_scale << SCALE_STD, SCALE_STD;
    this->theta_x.push_back(theta_x_scale);
    this->max_height = 100;
    this->max_width = 40;
    this->min_height = 1e6;
    this->min_width = 1e6;
    this->max_x = 0;
    this->max_y = 0;
    this->min_x = 1e6;
    this->min_y = 1e6;
    this->rng(12345);
    this->verbose = verbose;
    this->initialized = false;
}

void PHDParticleFilter::initialize(Mat& current_frame, vector<Rect> preDetections) {
    if(preDetections.size() > 0){
        this->img_size = current_frame.size();
        normal_distribution<double> position_random_x(0.0,theta_x.at(0)(0));
        normal_distribution<double> position_random_y(0.0,theta_x.at(0)(1));
        normal_distribution<double> scale_random_width(0.0,theta_x.at(1)(0));
        normal_distribution<double> scale_random_height(0.0,theta_x.at(1)(1));
        this->states.clear();
        this->weights.clear();
        this->particles_batch = (int)this->n_particles;
        double weight = (double)1.0f/this->n_particles;
    
        vector<Rect> detections;

#ifdef WITH_NMS
        nms(preDetections, detections, 0.5, 0);
#else
        detections = preDetections;
#endif

        for(size_t j = 0; j < detections.size(); j++){
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
                float _dy = position_random_y(this->generator);
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
        
        for (size_t i = 0; i < detections.size(); ++i)
        {
            Target target;
            target.label = i;
            target.color = Scalar(this->rng.uniform(0,255), this->rng.uniform(0,255), this->rng.uniform(0,255));
            target.bbox = detections.at(i);
            this->tracks.push_back(target);
            this->current_labels.push_back(i);
            this->labels.push_back(i);
        }
    
        this->initialized = true;
    }
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
        
        for (size_t i = 0; i < this->states.size(); i++){
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
                tmp_weights.push_back(SURVIVAL_RATE * this->weights.at(i));
            }
        }

        double lambda_birth = this->img_size.area() * BIRTH_RATE;
        /*poisson_distribution<int> birth_num(lambda_birth);
        int J_k = birth_num(this->generator);*/
        if(this->birth_model.size() > 0){
            for (size_t j = 0; j < this->birth_model.size(); j++){
                for (int k = 0; k < this->particles_batch; k++){
                    particle state;
                    state.width = this->birth_model[j].width + scale_random_width(this->generator);
                    state.height = this->birth_model[j].height + scale_random_height(this->generator);
                    state.x = this->birth_model[j].x + position_random_x(this->generator);
                    state.y = this->birth_model[j].y + position_random_y(this->generator);
                    Rect box(state.x, state.y, state.width, state.height);
                    tmp_new_states.push_back(state);
                    tmp_weights.push_back((double)lambda_birth/(this->birth_model.size() * this->particles_batch));
                }
            }
        }
        //cout << "new particles weight: " << (double)lambda_birth/(this->birth_model.size() * this->particles_batch) << endl;

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

void PHDParticleFilter::draw_particles(Mat& image, Scalar color = Scalar(0, 255, 255)){
    for (size_t i = 0; i < this->states.size(); i++){
        particle state = this->states[i];
        Point pt1, pt2;
        pt1.x = cvRound(state.x);
        pt1.y = cvRound(state.y);
        pt2.x = cvRound(state.x + state.width);
        pt2.y = cvRound(state.y + state.height);
        rectangle( image, pt1, pt2, color, 2, LINE_AA );
    }
}

void PHDParticleFilter::update(Mat& image, vector<Rect> preDetections)
{
    if(preDetections.size() > 0){
        vector<Rect> detections;

#ifdef WITH_NMS
        nms(preDetections, detections, 0.5, 0);
#else       
        detections = preDetections;
#endif

        this->birth_model.clear();
        vector<double> tmp_weights;
        MatrixXd observations = MatrixXd::Zero(detections.size(), 4);
        
        double clutter_prob = (double)CLUTTER_RATE/this->img_size.area();
        for (size_t j = 0; j < detections.size(); j++){
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
        
        if(this->verbose){
            Scalar phd_estimate = sum(this->weights);
            cout << "Update target number: "<< cvRound(phd_estimate[0]) << endl;
        }
        
        this->weights.swap(tmp_weights);
        resample();
        
        tmp_weights.clear();
    }
    /*cout << "---------------------------------------------------" << endl;
    cout << "track: " << this->tracks.size() << " | detections: " << detections.size() << " | states: " << this->states.size() << endl;*/
}

void PHDParticleFilter::resample(){
    int L_k = this->states.size();
    vector<double> cumulative_sum(L_k);
    vector<double> normalized_weights(L_k);
    vector<double> log_weights(L_k);
    vector<double> squared_normalized_weights(L_k);
    uniform_real_distribution<double> unif_rnd(0.0,1.0);
    for (int i = 0; i < L_k; i++) {
        log_weights[i] = log(this->weights[i]);
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
    Scalar phd_estimate = sum(this->weights);
    int N_k = cvRound(this->particles_batch * phd_estimate[0]);
    //cout << "Resampled target number: " << cvRound(phd_estimate[0]) << endl;
    //cout << "N_k: " << N_k << endl;

    vector<particle> tmp_new_states;
    vector<double> tmp_weights;
    for (int i = 0; i < N_k; i++) {
        double uni_rand = unif_rnd(this->generator);
        vector<double>::iterator pos = lower_bound(cumulative_sum.begin(), cumulative_sum.end(), uni_rand);
        int ipos = distance(cumulative_sum.begin(), pos);
        particle state = this->states[ipos];
        tmp_new_states.push_back(state);
        tmp_weights.push_back((double)1.0f/this->particles_batch);
    }
    this->states.swap(tmp_new_states);
    this->weights.swap(tmp_weights);
    tmp_new_states.clear();
    tmp_weights.clear();

    cumulative_sum.clear();
    squared_normalized_weights.clear();
}

vector<Target> PHDParticleFilter::estimate(Mat& image, bool draw){
    Scalar phd_estimate = sum(this->weights);
    //cout << "Estimated target number : "<< cvRound(phd_estimate[0]) << endl;
    int num_targets = cvRound(phd_estimate[0]);
    vector<Target> new_tracks;
    if(num_targets > 0)
    {
        /********************** OpenCV EM **********************/
        Mat data, em_labels, emMeans;
        data = Mat::zeros((int)this->states.size(),4, CV_64F);
        for (size_t j = 0; j < this->states.size(); j++){
            data.at<double>(j,0) = this->states[j].x;
            data.at<double>(j,1) = this->states[j].y;
            data.at<double>(j,2) = this->states[j].width;
            data.at<double>(j,3) = this->states[j].height;
        }
        Ptr<cv::ml::EM> em_model = cv::ml::EM::create();
        em_model->setClustersNumber(num_targets);
        em_model->setCovarianceMatrixType(cv::ml::EM::COV_MAT_DIAGONAL);
        em_model->setTermCriteria(TermCriteria(TermCriteria::COUNT, 10, 0.1));
        em_model->trainEM(data, noArray(), em_labels, noArray());
        emMeans = em_model->getMeans();
        
        int label = 0;
        for (int i = 0; i < emMeans.rows; ++i)
        {
            Target target;
            target.color = Scalar(this->rng.uniform(0,255), this->rng.uniform(0,255), this->rng.uniform(0,255));
            target.bbox = Rect(emMeans.at<double>(i,0), emMeans.at<double>(i,1), emMeans.at<double>(i,2), emMeans.at<double>(i,3));
            
            while( (find(this->labels.begin(), this->labels.end(), label) != this->labels.end()) ) label++;
            target.label = label;
            this->current_labels.push_back(label);
            label++;
            new_tracks.push_back(target);
        }
        /*******************************************************/
        
        if (this->tracks.size() > 0)
        {
            hungarian_problem_t p;
            int **m = Utils::compute_cost_matrix(this->tracks, new_tracks);// this->tracks --> this-states
            hungarian_init(&p, m, this->tracks.size(), new_tracks.size(), HUNGARIAN_MODE_MINIMIZE_COST);// this->tracks.size --> this->states().size()
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

        /*cout << endl << "labels" << endl;
        for(size_t i = 0; i < this->labels.size(); i++) cout << this->labels.at(i) << " ";*/

        if(this->verbose){
            cout << "estimated target num: " << cvRound(phd_estimate[0]) << endl;
        }

        new_tracks.clear();

        if (draw)
        {
            for (size_t i = 0; i < this->tracks.size(); ++i)
            {
                rectangle(image, this->tracks.at(i).bbox, this->tracks.at(i).color, 2, LINE_AA);
            }
        }
    }

    return this->tracks;
}
