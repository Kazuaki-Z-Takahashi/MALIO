
#include <algorithm>
#include <numeric>
#include "op_i_top.hpp"
#include "misc.hpp"

map<string, double> op_i_top::calc_i_wrapper(const vector<Vector3>& coord,
                                             const vector<Vector3>& direct,
                                             const Vector3& box_length,
                                             const vector<int>& neighbor_list_ii,
                                             int i_i,
                                             const I_SETTINGS& settings)
{
  map<string, double> op_temp;
  
  for(int o_k : settings.oi_oj_ok) {
    for(int o_j : settings.oi_oj_ok) {
      for(int o_i : settings.oi_oj_ok) {
        for(int o_f : settings.o_factor) {
          double sum_cos = 0.0;
          Vector3 x_i = misc::calc_head_coordinate(coord, direct, i_i, o_f, o_i);

          int N = neighbor_list_ii.size();
          for( int i_2 = 0; i_2 < N - 1; ++i_2) {
            int i_j = neighbor_list_ii[i_2];
            Vector3 x_j = misc::calc_head_coordinate(coord, direct, i_j, o_f, o_j);
            for(int i_3 = i_2 + 1; i_3 < N; ++i_3) {
              int i_k = neighbor_list_ii[i_3];
              Vector3 x_k = misc::calc_head_coordinate(coord, direct, i_k, o_f, o_k);
            
              Vector3 x_i_j = misc::calc_delta(x_j, x_i, box_length);
              Vector3 x_i_k = misc::calc_delta(x_k, x_i, box_length);
              
              double theta = 0.0;
              try {
                theta = misc::angle(x_i_j, x_i_k);
              }
              catch(...) {
              }
              if( theta >= M_PI )
                theta -= M_PI;

              double d = cos(theta) + 1.0/3.0;
              sum_cos += d * d;
            }
          }
          double op_i = 1.0 - (3.0 / 8.0) * sum_cos;

          string name = misc::naming_c(0, o_f, o_i, o_j, o_k);
          op_temp[name] = op_i;
        }
      }
    }
  }

  return op_temp;
}


map<string, vector<double>> op_i_top::top_order_parameter(const vector<Vector3>& coord,
                                                          const vector<Vector3>& direct,
                                                          const Vector3& box_length,
                                                          const I_SETTINGS& settings,
                                                          const vector<vector<int>>& neighbor_list)
{
  vector<map<string, double>> op_val_temp;

  int n_coord = coord.size();
  
  for(int i = 0; i < n_coord; ++i) {
    map<string, double> op_temp = calc_i_wrapper(coord, direct, box_length, neighbor_list[i], i, settings);
    op_val_temp.push_back(op_temp);
  }

  map<string, vector<double>> op_value = misc::data_num_name_to_data_name_num(op_val_temp, n_coord);

  // neighbor histogram averaging
  
  for( int o_k : settings.oi_oj_ok ) {
    for( int o_j : settings.oi_oj_ok ) {
      for( int o_i : settings.oi_oj_ok ) {
        for( int o_f : settings.o_factor ) {
          for(int a_t = 0; a_t < settings.ave_times; ++a_t) {
            string name = misc::naming_c(a_t + 1, o_f, o_i, o_j, o_k);
            string name_old = misc::naming_c(a_t, o_f, o_i, o_j, o_k);
            op_value[name] = misc::v_neighb_ave(neighbor_list, op_value[name_old]);
          }
        }
      }
    }
  }

  return op_value;
}

