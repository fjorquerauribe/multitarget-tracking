#include "utils.hpp"

int** Utils::compute_cost_matrix(vector<Target> tracks, vector<Target> new_tracks, double Ql, double Qs){
	double** cost_matrix = new double*[tracks.size()];
	int** norm_cost_matrix = new int*[tracks.size()];
	double max_cost=0.0f,min_cost=1000000.0f;
	for (size_t i = 0; i < tracks.size(); ++i)
	{
		cost_matrix[i] = new double[new_tracks.size()];
		Point2f pointTracked(tracks.at(i).bbox.x + tracks.at(i).bbox.width/2, tracks.at(i).bbox.y + tracks.at(i).bbox.height/2);
		Point2f scaleTracked(tracks.at(i).bbox.width, tracks.at(i).bbox.height);
		for (size_t j = 0; j < new_tracks.size(); ++j)
		{
			Point2f pointEstimated(new_tracks.at(j).bbox.x + new_tracks.at(j).bbox.width/2,
			new_tracks.at(j).bbox.y + new_tracks.at(j).bbox.height/2);
			Point2f scaleEstimated( new_tracks.at(j).bbox.width,new_tracks.at(j).bbox.height);
			double feature_cost = 1.0 + (tracks.at(i).feature - new_tracks.at(j).feature).squaredNorm();
			double position_cost = 1.0 + norm(pointTracked - pointEstimated)/Ql;
			double scale_cost = 1.0 + norm(scaleTracked - scaleEstimated)/Qs;
			//cout << "featCost: " << feature_cost << " | position_cost: " << position_cost << " | scale_cost: " << scale_cost << endl;
			double cost = position_cost * scale_cost * feature_cost;
			cost_matrix[i][j] = cost;
			max_cost = max(max_cost,cost);
			min_cost = min(min_cost,cost);
		}
	}
	for (size_t i = 0; i < tracks.size(); ++i)
	{
		norm_cost_matrix[i] = new int[new_tracks.size()];
		for (size_t j = 0; j < new_tracks.size(); ++j)
		{
			double x_std =((int)new_tracks.size()>1) ? (cost_matrix[i][j]-min_cost)/(max_cost-min_cost) : cost_matrix[i][j];
			norm_cost_matrix[i][j] = 100*x_std;
		}
	}
	return norm_cost_matrix;
}

int** Utils::compute_overlap_matrix(vector<Target> tracks, vector<Target> new_tracks){
	int** cost_matrix = new int*[tracks.size()];
	for(size_t i = 0; i < tracks.size(); i++){
		cost_matrix[i] = new int[new_tracks.size()];
		Rect current_track=tracks[i].bbox;
		for(size_t j = 0; j < new_tracks.size(); j++){
			Rect new_track=new_tracks[j].bbox;
			double Intersection = (double)(current_track & new_track).area();
			double Union=(double)current_track.area()+(double)new_track.area()-Intersection;
			cost_matrix[i][j] = 100*Intersection/Union;
			//cout << "cost i: " << i << ", j : " << j << ", c : " << cost_matrix[i][j] << endl; 
		}
	}
	return cost_matrix;
}

void Utils::detections_quality(VectorXd &detections_weights, vector<Rect> detections, vector<Target> tracks, 
	VectorXd &contain, double overlap_threshold, double lambda)
{
	contain = VectorXd::Zero(detections.size());
	detections_weights = (detections_weights.array() + 0.7805) / (138.92);

	for(size_t i = 0; i < detections.size(); i++){
		for(size_t j = 0; j < tracks.size(); j++){
			if( (double((detections[i] & tracks[j].bbox).area())/detections[i].area()) > overlap_threshold ){
				contain(i)++;
			}
		}
	}
	
	MatrixXd penalty = contain.array().exp().pow(lambda);
	float alpha = 1.0, beta = 1.0;
	
	detections_weights = detections_weights.cwiseProduct(penalty);
	//detections_weights = detections_weights.array() / detections_weights.maxCoeff();
	detections_weights = detections_weights.array() + 1;
	detections_weights = detections_weights.array().log() / log(10);
	detections_weights = alpha * detections_weights.array() + beta;
	detections_weights = detections_weights.array().square();
	
}