
#include <algorithm>
#include "op_a_cnp.hpp"
#include "misc.hpp"

map<string, double> op_a_cnp::calc_cnp_wrapper(const vector<Vector3>& coord,
                                               const vector<Vector3>& direct,
                                               const Vector3& box_length,
                                               const vector<int>& neighbor_list_ii,
                                               int i_i,
                                               const A_SETTINGS& settings)
{
  map<string, double> op_temp;

  const vector<int>& m_nei = settings.m_in_A;
  
  int max_m = *max_element(begin(m_nei), end(m_nei));
  
  for(int o_k : settings.oi_oj_ok) {
    for(int o_j : settings.oi_oj_ok) {
      for(int o_i : settings.oi_oj_ok) {
        for(int o_f : settings.o_factor) {
          Vector3 x_i = misc::calc_head_coordinate(coord, direct, i_i, o_f, o_i);
          vector<vector<int>> nei_ij = misc::gen_neighbor_ijk(coord, direct, box_length, neighbor_list_ii,
                                                              x_i, o_f, o_j, o_k, max_m);

          for( int now_m : m_nei ) {
            for( string op_type : settings.op_types ) {

              Vector3 sum_vec_m;
              double sum_r = 0.0;
              for( int now_j = 0; now_j < (int)neighbor_list_ii.size(); ++now_j ) {
                int i_j = neighbor_list_ii[now_j];
                Vector3 x_j = misc::calc_head_coordinate(coord, direct, i_j, o_f, o_j);

                Vector3 sum_vec;
                for( int now_k = 0; now_k < now_m; ++now_k) {
                  if( now_k >= (int)nei_ij[now_j].size() )
                    continue;
                  int i_k = nei_ij[now_j][now_k];
                  Vector3 x_k = misc::calc_head_coordinate(coord, direct, i_k, o_f, o_k);
                  if( op_type == "A" ) {
                    Vector3 r_ik = misc::calc_delta(x_k, x_i, box_length);
                    Vector3 r_jk = misc::calc_delta(x_k, x_j, box_length);
                    sum_vec += r_ik + r_jk;
                  }
                  else if( op_type == "P" || op_type == "N" ) {
                    Vector3 r_ij = misc::calc_delta(x_j, x_i, box_length);
                    Vector3 r_kj = misc::calc_delta(x_j, x_k, box_length);
                    sum_vec += r_ij + r_kj;
                  }
                }

                if( op_type == "A" || op_type == "P" ) {
                  sum_r += sum_vec * sum_vec;
                }
                else if( op_type == "N" ) {
                  sum_vec_m += sum_vec;
                }
              }
              
              if( op_type == "N" ) {
                sum_r = sum_vec_m * sum_vec_m;
              }
                  
              double now_op = sum_r / double(neighbor_list_ii.size());
              string name = misc::naming_a(0, op_type, now_m, o_f, o_i, o_j, o_k);
              op_temp[name] = now_op;
            }
          }
        }
      }
    }
  }

  return op_temp;
}


map<string, vector<double>> op_a_cnp::cnp_order_parameter(const vector<Vector3>& coord,
                                                          const vector<Vector3>& direct,
                                                          const Vector3& box_length,
                                                          const A_SETTINGS& settings,
                                                          const vector<vector<int>>& neighbor_list)
{
  vector<map<string, double>> op_temp;

  int n_coord = coord.size();
  
  for(int i = 0; i < n_coord; ++i) {
    map<string, double> op_temp_ = calc_cnp_wrapper(coord, direct, box_length, neighbor_list[i], i, settings);
    op_temp.push_back(op_temp_);
  }

  map<string, vector<double>> op_value = misc::data_num_name_to_data_name_num(op_temp, n_coord);

  for( int a_t = 0; a_t < settings.ave_times; ++a_t ) {
    for( int o_k : settings.oi_oj_ok ) {
      for( int o_j : settings.oi_oj_ok ) {
        for( int o_i : settings.oi_oj_ok ) {
          for( int o_f : settings.o_factor ) {
            for( int m : settings.m_in_A ) {
              for( string op_type : settings.op_types ) {
                string name = misc::naming_a(a_t + 1, op_type, m, o_f, o_i, o_j, o_k);
                string name_old = misc::naming_a(a_t, op_type, m, o_f, o_i, o_j, o_k);
                op_value[name] = misc::v_neighb_ave(neighbor_list, op_value[name_old]);
              }
            }
          }
        }
      }
    }
  }
  
  return op_value;
}

