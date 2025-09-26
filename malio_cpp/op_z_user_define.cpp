
#include "op_z_user_define.hpp"

map<string, vector<double>> op_z_user_define::user_define_parameter(const vector<Vector3>& coord,
                                                                    const vector<Vector3>& direct,
                                                                    const Vector3& box_length,
                                                                    const vector<vector<int>>& neighbor_list)
{
  int n_coord = coord.size();
  
  vector<double> data_list;
  for(int i_i = 0; i_i < n_coord; ++i_i) {
    double sum_distance = 0.0;
    for( int i_j = 0; i_j < (int)neighbor_list[i_i].size(); ++i_j) {
      Vector3 x_i_j = coord[i_i] - coord[i_j];
      sum_distance += x_i_j.Norm();
    }
    data_list.push_back(sum_distance);
  }

  map<string, vector<double>> op_data;
  op_data["user_define"] = data_list;

  return op_data;
}



