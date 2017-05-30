#include "utils.hpp"

int** Utils::compute_cost_matrix(vector<Target> tracks, vector<Target> new_tracks){
	int** cost_matrix = new int*[tracks.size()];
	for (size_t i = 0; i < tracks.size(); ++i)
	{
		cost_matrix[i] = new int[new_tracks.size()];
		Point2f pointTracked(tracks.at(i).bbox.x + tracks.at(i).bbox.width/2, tracks.at(i).bbox.y + tracks.at(i).bbox.height/2);
		for (size_t j = 0; j < new_tracks.size(); ++j)
		{
			Point2f pointEstimated(new_tracks.at(j).bbox.x + new_tracks.at(j).bbox.width/2,
			 new_tracks.at(j).bbox.y + new_tracks.at(j).bbox.height/2);
			cost_matrix[i][j] = norm(pointTracked - pointEstimated);
		}
	}
	return cost_matrix;
}