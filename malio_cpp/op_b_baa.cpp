
#include <algorithm>
#include "op_b_baa.hpp"
#include "misc.hpp"

map<string, double> op_b_baa::calc_baa_wrapper(const vector<Vector3>& coord,
                                               const vector<Vector3>& direct,
                                               const Vector3& box_length,
                                               const vector<int>& neighbor_list_ii,
                                               int i_i,
                                               const B_SETTINGS& settings)
{
  map<string, double> b_list;

  for(int o_k : settings.oi_oj_ok) {
    for(int o_j : settings.oi_oj_ok) {
      for(int o_i : settings.oi_oj_ok) {
        for(int o_f : settings.o_factor) {
          for(int n_pow : settings.n) {
            for(double phi : settings.phi) {
              for(int m_fac : settings.m) {

                Vector3 x_i = misc::calc_head_coordinate(coord, direct, i_i, o_f, o_i);

                double op_temp = 0.0;
                for( int i_2 = 0; i_2 < (int)neighbor_list_ii.size() - 1; ++i_2) {
                  int i_j = neighbor_list_ii[i_2];
                  Vector3 x_j = misc::calc_head_coordinate(coord, direct, i_j, o_f, o_j);
                  for( int i_3 = i_2 + 1; i_3 < (int)neighbor_list_ii.size(); ++i_3) {
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

                    op_temp += pow(cos(m_fac * theta + phi), n_pow);
                  }
                }

                double n_1 = double(neighbor_list_ii.size());
                op_temp /= (n_1 * (n_1 - 1)) / 2;

                string name = misc::naming_b(0, m_fac, phi, n_pow, o_f, o_i, o_j, o_k);
                b_list[name] = op_temp;
              }
            }
          }
        }
      }
    }
  }

  return b_list;
}


map<string, vector<double>> op_b_baa::baa_order_parameter(const vector<Vector3>& coord,
                                                          const vector<Vector3>& direct,
                                                          const Vector3& box_length,
                                                          const B_SETTINGS& settings,
                                                          const vector<vector<int>>& neighbor_list)
{
  vector<map<string, double>> op_temp;

  int n_coord = coord.size();
  
  for(int i = 0; i < n_coord; ++i) {
    map<string, double> op_temp_ = calc_baa_wrapper(coord, direct, box_length, neighbor_list[i], i, settings);
    op_temp.push_back(op_temp_);
  }

  map<string, vector<double>> op_value = misc::data_num_name_to_data_name_num(op_temp, n_coord);

  for( int a_t = 0; a_t < settings.ave_times; ++a_t ) {
    for( int m_fac : settings.m ) {
      for( double phi : settings.phi ) {
        for( int n_pow : settings.n ) {
          for( int o_f : settings.o_factor ) {
            for( int o_i : settings.oi_oj_ok ) {
              for( int o_j : settings.oi_oj_ok ) {
                for( int o_k : settings.oi_oj_ok ) {
                  string name = misc::naming_b(a_t + 1, m_fac, phi, n_pow, o_f, o_i, o_j, o_k);
                  string name_old = misc::naming_b(a_t, m_fac, phi, n_pow, o_f, o_i, o_j, o_k);
                  op_value[name] = misc::v_neighb_ave(neighbor_list, op_value[name_old]);
                }
              }
            }
          }
        }
      }
    }
  }

  return op_value;
}

