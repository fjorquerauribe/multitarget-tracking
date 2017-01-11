/**
 * @file particle_phd_filter.cpp
 * @brief particle filter
 * @author Sergio Hernandez
 */
#include "particle_filter.hpp"

#ifndef PARAMS
const float POS_STD=1.0;
const float SCALE_STD=1.0;
const float THRESHOLD=1000;
const float SURVIVAL_RATE=0.9;
const float CLUTTER_RATE=1;
const float BIRTH_RATE=3e-6;
const float DETECTION_RATE=0.9;
const float POSITION_LIKELIHOOD_STD=10.0;
#endif 

particle_filter::particle_filter() {
}

particle_filter::~particle_filter() {
    states.clear();
    weights.clear();
}

bool particle_filter::is_initialized() {
    return initialized;
}

particle_filter::particle_filter(int _n_particles) {
    states.clear();
    weights.clear();
    n_particles = _n_particles;
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed1);
    theta_x.clear();
    RowVectorXd theta_x_pos(2);
    theta_x_pos << POS_STD,POS_STD;
    theta_x.push_back(theta_x_pos);
    RowVectorXd theta_x_scale(2);
    theta_x_scale << SCALE_STD,SCALE_STD;
    theta_x.push_back(theta_x_scale);
    max_height=100;
    max_width=40;
    min_height=1e6;
    min_width=1e6;
    max_x=0;
    max_y=0;
    min_x=1e6;
    min_y=1e6;
    initialized=false;
    //initParallel();
    //setNbThreads(4);
    //int nthreads = Eigen::nbThreads( );
    //std::cout << "THREADS = " << nthreads <<std::ends; // returns '1'
}

void particle_filter::initialize(Mat& current_frame, vector<Rect> detections) {
    im_size=current_frame.size();
    normal_distribution<double> position_random_x(0.0,theta_x.at(0)(0));
    normal_distribution<double> position_random_y(0.0,theta_x.at(0)(1));
    normal_distribution<double> scale_random_width(0.0,theta_x.at(1)(0));
    normal_distribution<double> scale_random_height(0.0,theta_x.at(1)(1));
    states.clear();
    weights.clear();
    double weight=(double)detections.size()/n_particles;
    particles_batch=n_particles/(int)detections.size();
    int remaining_batch=n_particles % (int)detections.size();
    for(unsigned int j=0;j<detections.size();j++){
        max_width=MAX(detections[j].width,max_width);
        max_height=MAX(detections[j].height,max_height);
        min_width=MIN(detections[j].width,min_width);
        min_height=MIN(detections[j].height,min_height);
        max_x=MAX(detections[j].x,max_x);
        max_y=MAX(detections[j].y,max_y);
        min_x=MIN(detections[j].x,min_x);
        min_y=MIN(detections[j].y,min_y);
        for (int i=0;i<particles_batch;i++){
            particle state;
            float _x,_y,_width,_height;
            float _dx=position_random_x(generator);
            float _dy=+position_random_y(generator);
            float _dw=0.0f;//scale_random_width(generator);
            float _dh=0.0f;//scale_random_height(generator);
            _x=MIN(MAX(cvRound(detections[j].x+_dx),0),im_size.width);
            _y=MIN(MAX(cvRound(detections[j].y+_dy),0),im_size.height);
            _width=MIN(MAX(cvRound(detections[j].width+_dw),0),im_size.width);
            _height=MIN(MAX(cvRound(detections[j].height+_dh),0),im_size.height);
            state.x=_x;
            state.y=_y;
            state.width=_width;
            state.height=_height;   
            state.scale=1.0; 
            states.push_back(state);
            weights.push_back(weight);    
        }
    }
    uniform_int_distribution<int> random_x(min_x,max_x);
    uniform_int_distribution<int> random_y(min_y,max_y);
    uniform_int_distribution<int> random_w(min_width,max_width);
    uniform_int_distribution<int> random_h(min_height,max_height);
    for (int i=0;i<remaining_batch;i++){
        particle state;
        state.width=random_w(generator);
        state.height=random_h(generator);
        state.x=cvRound(random_x(generator));
        state.y=cvRound(random_y(generator));  
        state.scale=1.0;         
        states.push_back(state);
        weights.push_back(1/100.f);    
    }    
    initialized=true;
    Scalar phd_estimate=sum(weights);
    cout << "initial estimated number : "<< phd_estimate[0] << endl; 
}

void particle_filter::predict(){
    normal_distribution<double> position_random_x(0.0,theta_x.at(0)(0));
    normal_distribution<double> position_random_y(0.0,theta_x.at(0)(1));
    normal_distribution<double> scale_random_width(0.0,theta_x.at(1)(0));
    normal_distribution<double> scale_random_height(0.0,theta_x.at(1)(1));
    uniform_real_distribution<double> unif(0.0,1.0);
    uniform_int_distribution<int> random_x(min_x,max_x);
    uniform_int_distribution<int> random_y(min_y,max_y);
    uniform_int_distribution<int> random_w(min_width,max_width);
    uniform_int_distribution<int> random_h(min_height,max_height);
    double lambda_birth=im_size.area()*BIRTH_RATE;
    poisson_distribution<int> birth_num(lambda_birth);
    int J_k=birth_num(generator);
    if(initialized==true){
        vector<particle> tmp_new_states;
        vector<double> tmp_weights;
        for (unsigned int i=0;i<states.size();i++){
            particle state=states[i];
            float _x,_y,_width,_height;
            float _dx=position_random_x(generator);
            float _dy=position_random_y(generator);
            float _dw=0.0f;//scale_random_width(generator);
            float _dh=0.0f;//scale_random_height(generator);
            _x=MIN(MAX(cvRound(state.x+_dx),0),im_size.width);
            _y=MIN(MAX(cvRound(state.y+_dy),0),im_size.height);
            _width=MIN(MAX(cvRound(state.width+_dw),0),im_size.width);
            _height=MIN(MAX(cvRound(state.height+_dh),0),im_size.height);
            if((_x+_width)<im_size.width && _x>0 && 
                (_y+_height)<im_size.height && _y>0 && 
                _width<im_size.width && _height<im_size.height && 
                _width>0 && _height>0 && unif(generator)<SURVIVAL_RATE){
                //&& unif(generator)<SURVIVAL_RATE
                state.x_p=state.x;
                state.y_p=state.y;
                state.width_p=state.width;
                state.height_p=state.height;       
                state.x=_x;
                state.y=_y;
                state.width=_width;
                state.height=_height;
                state.scale_p=state.scale;
                state.scale=2*state.scale-state.scale_p+scale_random_width(generator);
                Rect box(state.x, state.y, state.width, state.height);
                tmp_new_states.push_back(state);
                tmp_weights.push_back(weights.at(i));
            }
        }
        for (int j=0;j<J_k;j++){
            for (int k=0;k<particles_batch;k++){
                particle state;
                state.width=cvRound(random_w(generator));
                state.height=cvRound(random_h(generator));
                state.x=cvRound(random_x(generator));
                state.y=cvRound(random_y(generator));
                Rect box(state.x, state.y, state.width, state.height);
                tmp_new_states.push_back(state);
                tmp_weights.push_back((double)J_k/n_particles);
            }
        }
        states.swap(tmp_new_states);
        weights.swap(tmp_weights);
        Scalar phd_estimate=sum(weights);
        tmp_new_states = vector<particle>();
        tmp_weights = vector<double>();
        cout << "predicted target number : "<< (int)phd_estimate[0] << endl; 
        cout << "predicted birth number : "<< J_k << endl; 
    }
}

void particle_filter::draw_particles(Mat& image, Scalar color=Scalar(0,255,255)){
    for (unsigned int i=0;i<states.size();i++){
        particle state=states[i];
        Point pt1,pt2;
        pt1.x=cvRound(state.x);
        pt1.y=cvRound(state.y);
        pt2.x=cvRound(state.x+state.width);
        pt2.y=cvRound(state.y+state.height);
        rectangle( image, pt1,pt2, color, 1, LINE_AA );
    }
}

void particle_filter::auxiliary(Mat& image, vector<Rect> detections)
{
    vector<double> aux_weights,tmp_weights;
    MatrixXd cov=POSITION_LIKELIHOOD_STD*POSITION_LIKELIHOOD_STD*MatrixXd::Identity(4, 4);
    MatrixXd observations=MatrixXd::Zero(detections.size(),4);
    for (unsigned int j=0;j<detections.size();j++){
        max_width=MAX(detections[j].width,max_width);
        max_height=MAX(detections[j].height,max_height);
        min_width=MIN(detections[j].width,min_width);
        min_height=MIN(detections[j].height,min_height);
        max_x=MAX(detections[j].x,max_x);
        max_y=MAX(detections[j].y,max_y);
        min_x=MIN(detections[j].x,min_x);
        min_y=MIN(detections[j].y,min_y);
        observations.row(j) << detections[j].x, detections[j].y, detections[j].width,detections[j].height;
    }
    for (int i=0;i<n_particles;i++){
        particle state=states[i];
        double weight=weights[i];
        VectorXd  mean(4);
        mean << state.x, state.y, state.width,state.height;
        MVNGaussian gaussian(mean,cov);
        VectorXd psi=DETECTION_RATE*weight*gaussian.log_likelihood(observations);
        double sumexp=0.0;
        double max_value = psi.maxCoeff();
        for (unsigned int j=0; j<detections.size(); j++) {
            sumexp+=exp(psi[j]-max_value);
        }
        double norm_const=max_value+log(sumexp);
        VectorXd phd_update= (psi.array()-norm_const-CLUTTER_RATE/(im_size.width*im_size.height)).exp().matrix();
        aux_weights.push_back(phd_update.array().sum()+(1-DETECTION_RATE)*weight);
    }
    //weights.swap(tmp_weights);
    //
    resample();
    predict();    
    for (int i=0;i<n_particles;i++){
        particle state=states[i];
        double weight=weights[i];
        VectorXd  mean(4);
        mean << state.x, state.y, state.width,state.height;
        MVNGaussian gaussian(mean,cov);
        VectorXd psi=DETECTION_RATE*weight*gaussian.log_likelihood(observations);
        double sumexp=0.0;
        double max_value = psi.maxCoeff();
        for (unsigned int j=0; j<detections.size(); j++) {
            sumexp+=exp(psi[j]-max_value);
        }
        double norm_const=max_value+log(sumexp);
        VectorXd phd_update= (psi.array()-norm_const-CLUTTER_RATE*im_size.area()).exp().matrix();
        tmp_weights.push_back(phd_update.array().sum()+(1-DETECTION_RATE)*weight);
    }
    for (unsigned int k=0; k<weights.size(); k++) {
        weights[k]=tmp_weights.at(k)/aux_weights.at(k);
    }
    tmp_weights.clear();
    aux_weights.clear();
    resample();
    
}

void particle_filter::update(Mat& image, vector<Rect> detections)
{
    if(detections.size()>0){
        vector<double> tmp_weights;
        MatrixXd cov=POSITION_LIKELIHOOD_STD*POSITION_LIKELIHOOD_STD*MatrixXd::Identity(4, 4);
        //cout << "detections : " << detections.size() << endl;
        MatrixXd observations=MatrixXd::Zero(detections.size(),4);
        //double clutter_prob=log(CLUTTER_RATE/im_size.area());
        for (unsigned int j=0;j<detections.size();j++){
            max_width=MAX(detections[j].width,max_width);
            max_height=MAX(detections[j].height,max_height);
            min_width=MIN(detections[j].width,min_width);
            min_height=MIN(detections[j].height,min_height);
            max_x=MAX(detections[j].x,max_x);
            max_y=MAX(detections[j].y,max_y);
            min_x=MIN(detections[j].x,min_x);
            min_y=MIN(detections[j].y,min_y);
            observations.row(j) << detections[j].x, detections[j].y, detections[j].width,detections[j].height;
        }
        for (unsigned int i=0;i<states.size();i++){
            //cout << "OBS,--------------------------------------------" << endl;
            //cout << observations << endl;
            particle state=states[i];
            double weight=weights[i];
            VectorXd  mean(4);
            //cout << "Mean" << endl;
            mean << state.x, state.y, state.width,state.height;
            //cout << mean.transpose() << endl;
            MVNGaussian gaussian(mean,cov);
            //cout << "likelihood" << endl;
            ArrayXd psi=log(DETECTION_RATE)+log(weight)+gaussian.log_likelihood(observations).array();
            //cout << psi.transpose() << endl;
            double logsumexp=0.0;
            double max_value = psi.maxCoeff();
            for (unsigned int j=0; j<detections.size(); j++) {
                logsumexp+=exp(psi[j]-max_value);
            }
            double norm_const=max_value+log(logsumexp);
            ArrayXd phd_update=(psi-norm_const).exp();
            tmp_weights.push_back( weight*phd_update.sum());
        }
        weights.swap(tmp_weights);
        Scalar phd_estimate=sum(weights);
        cout << "Updated target number : "<< (int)phd_estimate[0] << endl; 
        resample();
        tmp_weights.clear();
    }
}


void particle_filter::resample(){
    int L_k=states.size();
    Scalar phd_estimate=sum(weights);
    vector<double> cumulative_sum(L_k);
    vector<double> normalized_weights(L_k);
    vector<double> new_weights(L_k);
    vector<double> squared_normalized_weights(L_k);
    uniform_real_distribution<double> unif_rnd(0.0,1.0); 
    double logsumexp=0.0;
    double max_value = *max_element(weights.begin(), weights.end());
    for (unsigned int i=0; i<weights.size(); i++) {
        new_weights[i]=exp(weights[i]-max_value);
        logsumexp+=new_weights[i];
    }
    double norm_const=max_value+log(logsumexp);
    for (unsigned int i=0; i<weights.size(); i++) {
        normalized_weights.at(i) = exp(weights.at(i)-norm_const);
    }
    for (unsigned int i=0; i<weights.size(); i++) {
        squared_normalized_weights.at(i)=normalized_weights.at(i)*normalized_weights.at(i);
        if (i==0) {
            cumulative_sum.at(i) = normalized_weights.at(i);
        } else {
            cumulative_sum.at(i) = cumulative_sum.at(i-1) + normalized_weights.at(i);
        }
    }
    Scalar sum_squared_weights=sum(squared_normalized_weights);
    //double marginal_likelihood=norm_const-log(n_particles); 
    double ESS=(1.0f/sum_squared_weights[0])/n_particles;
    cout << "ESS:"<< ESS << endl;
    if(isless(ESS,(float)THRESHOLD)){
        vector<particle> new_states;
        for (int i=0; i<L_k; i++) {
            double uni_rand = unif_rnd(generator);
            vector<double>::iterator pos = lower_bound(cumulative_sum.begin(), cumulative_sum.end(), uni_rand);
            int ipos = distance(cumulative_sum.begin(), pos);
            particle state=states[ipos];
            new_states.push_back(state);
            weights.at(i)=double(phd_estimate[0])/L_k;
        }
        states.swap(new_states);
    }
    cumulative_sum.clear();
    squared_normalized_weights.clear();
}

vector<Rect> particle_filter::estimate(Mat& image,bool draw=false){
    vector<Target> new_tracks;
    Scalar phd_estimate=sum(weights);
    int num_targets=(int)phd_estimate[0];
    vector<Rect> estimates(num_targets);
    MatrixXd data((int)states.size(),4);
    for (unsigned int j=0;j<states.size();j++){
        data.row(j) << states[j].x, states[j].y, states[j].width,states[j].height;
    }
    EM mixture(data,num_targets);
    MatrixXd cost_matrix=MatrixXd::Zero(num_targets,tracks.size());
    //double result=mixture.fit(10);
    vector<VectorXd> eigen_estimates=mixture.getMeans();
    for(unsigned int k=0;k<eigen_estimates.size();k++){
        VectorXd vec=eigen_estimates[k];
        for(unsigned int n=0;n<tracks.size();n++){
            VectorXd eigen_track;
            eigen_track << tracks[n].bbox.x, tracks[n].bbox.y, tracks[n].bbox.width,tracks[n].bbox.height;
        }
        Point pt1,pt2;
        pt1.x=cvRound(vec(0));
        pt1.y=cvRound(vec(1));
        float _width=cvRound(vec(2));
        float _height=cvRound(vec(3));
        pt2.x=cvRound(pt1.x+_width);
        pt2.y=cvRound(pt1.y+_height); 
        if(vec[0]<im_size.width && vec[0]>=0 && vec[1]<im_size.height && vec[1]>=0){
               if(draw) rectangle( image, pt1,pt2, Scalar(0,0,255), 2, LINE_AA );
            Rect estimate=Rect(pt1.x,pt1.y,_width,_height);
            estimates.push_back(estimate);
            //Target target;
            //target.bbox=estimate;
            //target.label=k;
            //tracks.push_back(target);
        }
    }
    return estimates;
}
