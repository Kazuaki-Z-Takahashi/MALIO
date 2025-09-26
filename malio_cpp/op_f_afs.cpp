
#include <algorithm>
#include "op_f_afs.hpp"
#include "misc.hpp"

map<string, double> op_f_afs::calc_f_wrapper(const vector<Vector3>& coord,
                                             const vector<Vector3>& direct,
                                             const Vector3& box_length,
                                             const vector<int>& neighbor_list_ii,
                                             int i_i,
                                             const F_SETTINGS& settings)
{
  map<string, double> op_temp;
  const vector<FUNC>& func = settings.func;
  int nf = func.size();
  
  for(int l : settings.l_list) {
    for(int f_2 = 0; f_2 < nf; ++f_2) {
      for(int f_1 = 0; f_1 < nf; ++f_1) {
        for(int o_k : settings.oi_oj_ok) {
          for(int o_j : settings.oi_oj_ok) {
            for(int o_i : settings.oi_oj_ok) {
              for(int o_f : settings.o_factor) {
                string name = misc::naming_f(0, o_f, o_i, o_j, o_k, f_1, f_2, l);

                Vector3 x_i = misc::calc_head_coordinate(coord, direct, i_i, o_f, o_i);

                int N = neighbor_list_ii.size();
                double d_sum = 0.0;
                for( int i_2 = 0; i_2 < N; ++i_2) {
                  int i_j = neighbor_list_ii[i_2];
                  Vector3 x_j = misc::calc_head_coordinate(coord, direct, i_j, o_f, o_j);
                  Vector3 x_i_j = misc::calc_delta(x_j, x_i, box_length);
                  double d_i_j = x_i_j.Norm();

                  for(int i_3 = i_2 + 1; i_3 < N; ++i_3) {
                    int i_k = neighbor_list_ii[i_3];
                    Vector3 x_k = misc::calc_head_coordinate(coord, direct, i_k, o_f, o_k);
                    Vector3 x_i_k = misc::calc_delta(x_k, x_i, box_length);
                    double d_i_k = x_i_k.Norm();

                    double theta = 0.0;
                    try {
                      theta = misc::angle(x_i_j, x_i_k);
                    }
                    catch(...) {
                    }
                    if( theta >= M_PI )
                      theta -= M_PI;
                    
                    d_sum += func[f_1](d_i_j) * func[f_2](d_i_k) * cos(l * theta);
                  }
                }

                if( N <= 1 )
                  op_temp[name] = 0;
                else
                  op_temp[name] = d_sum / (N * (N-1) / 2);
                  
              }
            }
          }
        }
      }
    }
  }

  return op_temp;
}


map<string, vector<double>> op_f_afs::afs_order_parameter(const vector<Vector3>& coord,
                                                          const vector<Vector3>& direct,
                                                          const Vector3& box_length,
                                                          const F_SETTINGS& settings,
                                                          const vector<vector<int>>& neighbor_list)
{
  vector<map<string, double>> op_val_temp;

  int n_coord = coord.size();
  
  for(int i = 0; i < n_coord; ++i) {
    map<string, double> op_temp = calc_f_wrapper(coord, direct, box_length, neighbor_list[i], i, settings);
    op_val_temp.push_back(op_temp);
  }

  map<string, vector<double>> op_value = misc::data_num_name_to_data_name_num(op_val_temp, n_coord);

  int nf = settings.func.size();
  
  for( int a_t = 0; a_t < settings.ave_times; ++a_t ) {
    for( int o_f : settings.o_factor ) {
      for( int o_i : settings.oi_oj_ok ) {
        for( int o_j : settings.oi_oj_ok ) {
          for( int o_k : settings.oi_oj_ok ) {
            for(int f_1 = 0; f_1 < nf; ++f_1) {
              for(int f_2 = 0; f_2 < nf; ++f_2) {
                for(int l : settings.l_list) {
                  string name = misc::naming_f(a_t + 1, o_f, o_i, o_j, o_k, f_1, f_2, l);
                  string name_old = misc::naming_f(a_t, o_f, o_i, o_j, o_k, f_1, f_2, l);
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

