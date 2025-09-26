
#include <algorithm>
#include "op_c_cpa.hpp"
#include "misc.hpp"

map<string, double> op_c_cpa::calc_cpa_wrapper(const vector<Vector3>& coord,
                                               const vector<Vector3>& direct,
                                               const Vector3& box_length,
                                               const vector<int>& neighbor_list_ii,
                                               int i_i,
                                               const C_SETTINGS& settings)
{
  const vector<int>& nei_ii = neighbor_list_ii;
                                                 
  map<string, double> op_temp;

  for(int o_k : settings.oi_oj_ok) {
    for(int o_j : settings.oi_oj_ok) {
      for(int o_i : settings.oi_oj_ok) {
        for(int o_f : settings.o_factor) {
          string name = misc::naming_c(0, o_f, o_i, o_j, o_k);
          Vector3 x_i = misc::calc_head_coordinate(coord, direct, i_i, o_f, o_i);

          int half_N = int(nei_ii.size() / 2);
          double sum_dist = 0.0;
          for( int i_2 = 0; i_2 < half_N; ++i_2) {
            int i_j = nei_ii[i_2];
            Vector3 x_j = misc::calc_head_coordinate(coord, direct, i_j, o_f, o_j);
            int i_k = misc::search_opposite_j_particle(coord, direct, nei_ii, x_i, i_j, x_j, box_length, o_f, o_k);

            Vector3 x_k = misc::calc_head_coordinate(coord, direct, i_k, o_f, o_k);
            Vector3 x_i_j = misc::calc_delta(x_j, x_i, box_length);
            Vector3 x_i_k = misc::calc_delta(x_k, x_i, box_length);
            Vector3 x_ij_ik = x_i_j + x_i_k;

            sum_dist += x_ij_ik * x_ij_ik;
          }

          op_temp[name] = sum_dist / half_N;
        }
      }
    }
  }

  return op_temp;
}


map<string, vector<double>> op_c_cpa::cpa_order_parameter(const vector<Vector3>& coord,
                                                          const vector<Vector3>& direct,
                                                          const Vector3& box_length,
                                                          const C_SETTINGS& settings,
                                                          const vector<vector<int>>& neighbor_list)
{
  vector<map<string, double>> op_temp;

  int n_coord = coord.size();
  
  for(int i = 0; i < n_coord; ++i) {
    map<string, double> op_temp_ = calc_cpa_wrapper(coord, direct, box_length, neighbor_list[i], i, settings);
    op_temp.push_back(op_temp_);
  }

  map<string, vector<double>> op_value = misc::data_num_name_to_data_name_num(op_temp, n_coord);

  for( int a_t = 0; a_t < settings.ave_times; ++a_t ) {
    for( int o_f : settings.o_factor ) {
      for( int o_i : settings.oi_oj_ok ) {
        for( int o_j : settings.oi_oj_ok ) {
          for( int o_k : settings.oi_oj_ok ) {
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

