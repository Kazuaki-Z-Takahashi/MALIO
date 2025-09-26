
#include <algorithm>
#include <numeric>
#include "op_h_aha.hpp"
#include "misc.hpp"

static vector<double> histogram_normalize(const vector<double>& hist)
{
  double sum_hist = accumulate(hist.begin(), hist.end(), 0.0);
  vector<double> hist_ = hist;
  for(int i = 0; i < (int)hist_.size(); ++i) {
    hist_[i] /= sum_hist;
  }
  return hist_;
}

map<string, vector<double>> op_h_aha::calc_h_wrapper(const vector<Vector3>& coord,
                                             const vector<Vector3>& direct,
                                             const Vector3& box_length,
                                             const vector<int>& neighbor_list_ii,
                                             int i_i,
                                             const H_SETTINGS& settings)
{
  map<string, vector<double>> op_temp;

  const vector<int>& h_num = settings.hist_num;

  for(int o_k : settings.oi_oj_ok) {
    for(int o_j : settings.oi_oj_ok) {
      for(int o_i : settings.oi_oj_ok) {
        for(int o_f : settings.o_factor) {
          // init histogram
          vector<vector<double>> hist_temp;  // hist_temp[bin_num][ibin]
          for(int i_k = 0; i_k < (int)h_num.size(); ++i_k) {
            vector<double> h_temp2(h_num[i_k], 0.0);
            hist_temp.push_back(h_temp2);
          }

          Vector3 x_i = misc::calc_head_coordinate(coord, direct, i_i, o_f, o_i);

          int n_list = neighbor_list_ii.size();

          for(int i_2 = 0; i_2 < n_list - 1; ++i_2) {
            int i_j = neighbor_list_ii[i_2];
            Vector3 x_j = misc::calc_head_coordinate(coord, direct, i_j, o_f, o_j);
            for(int i_3 = i_2 + 1; i_3 < n_list; ++i_3) {
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

              for(int i_k = 0; i_k < (int)h_num.size(); ++i_k) {
                int h_i = h_num[i_k];
                int now_i = int(h_i * theta / M_PI);
                hist_temp[i_k][now_i] += 1.0 / double(n_list * (n_list - 1) / 2);
              }
            }
          }
          
          for(int i_k = 0; i_k < (int)h_num.size(); ++i_k) {
            string name = misc::naming_h(0, 0, h_num[i_k], o_f, o_i, o_j, o_k);
            op_temp[name] = hist_temp[i_k];
          }

        }
      }
    }
  }

  return op_temp;
}


map<string, vector<double>> op_h_aha::aha_order_parameter(const vector<Vector3>& coord,
                                                          const vector<Vector3>& direct,
                                                          const Vector3& box_length,
                                                          const H_SETTINGS& settings,
                                                          const vector<vector<int>>& neighbor_list)
{
  vector<map<string, vector<double>>> op_val_temp;

  int n_coord = coord.size();
  
  for(int i = 0; i < n_coord; ++i) {
    map<string, vector<double>> op_temp = calc_h_wrapper(coord, direct, box_length, neighbor_list[i], i, settings);
    op_val_temp.push_back(op_temp);
  }

  map<string, vector<vector<double>>> h_hist = misc::data_num_name_to_data_name_num(op_val_temp, n_coord);

  const vector<int>& h_num = settings.hist_num;
  int b_times = settings.b_in_H;
  
  for( int o_k : settings.oi_oj_ok ) {
    for( int o_j : settings.oi_oj_ok ) {
      for( int o_i : settings.oi_oj_ok ) {
        for( int o_f : settings.o_factor ) {
          for(int b_t = 0; b_t < b_times; ++b_t) {
            for(int i_k = 0; i_k < (int)h_num.size(); ++i_k) {
              string name = misc::naming_h(0, b_t + 1, h_num[i_k], o_f, o_i, o_j, o_k);
              string name_old = misc::naming_h(0, b_t, h_num[i_k], o_f, o_i, o_j, o_k);
              h_hist[name] = misc::v_neighb_ave(neighbor_list, h_hist[name_old]);
            }
          }
        }
      }
    }
  }

  // FFT
  map<string, vector<vector<double>>> h_data_part_nu;
  for( int o_k : settings.oi_oj_ok ) {
    for( int o_j : settings.oi_oj_ok ) {
      for( int o_i : settings.oi_oj_ok ) {
        for( int o_f : settings.o_factor ) {
          for(int b_t = 0; b_t < b_times + 1; ++b_t) {
            for(int i_k = 0; i_k < (int)h_num.size(); ++i_k) {
              string name = misc::naming_h(0, b_t, h_num[i_k], o_f, o_i, o_j, o_k);
              vector<vector<double>> g_list;
              for(int i_i = 0; i_i < n_coord; ++i_i) {
                vector<double> g_power = misc::fft_power(histogram_normalize(h_hist[name][i_i]));
                vector<double> power;
                for(int inu : settings.nu) {
                  if( 0 <= inu && inu < (int)g_power.size() ) {
                    power.push_back(g_power[inu]);
                  }
                  else {
                    power.push_back(0.0);
                  }
                }
                g_list.push_back(power);
              }
              h_data_part_nu[name] = g_list;
            }
          }
        }
      }
    }
  }
  
  map<string, vector<double>> op_value;
                
  for( int o_k : settings.oi_oj_ok ) {
    for( int o_j : settings.oi_oj_ok ) {
      for( int o_i : settings.oi_oj_ok ) {
        for( int o_f : settings.o_factor ) {
          for(int b_t = 0; b_t < b_times + 1; ++b_t) {
            for(int i_k = 0; i_k < (int)h_num.size(); ++i_k) {
              for(int i_l = 0; i_l < (int)settings.nu.size(); ++i_l) {
                int inu = settings.nu[i_l];
                string name = misc::naming_h(0, b_t, h_num[i_k], o_f, o_i, o_j, o_k);
                string name_h = name + "_nu=" + to_string(inu);
                vector<double> v(n_coord);
                for(int i_i = 0; i_i < n_coord; ++i_i) {
                  v[i_i] = h_data_part_nu[name][i_i][i_l];
                }
                op_value[name_h] = v;
              }
            }
          }
        }
      }
    }
  }
  
  for(int a_t = 0; a_t < settings.ave_times; ++a_t) {
    for(int b_t = 0; b_t < b_times + 1; ++b_t) {
      for(int i_k = 0; i_k < (int)h_num.size(); ++i_k) {
        for( int dist_layers : settings.oi_oj_ok ) {
          for( int o_j : settings.oi_oj_ok ) {
            for( int o_i : settings.oi_oj_ok ) {
              for( int o_f : settings.o_factor ) {
                for(int i_l = 0; i_l < (int)settings.nu.size(); ++i_l) {
                  int inu = settings.nu[i_l];
                  string name = misc::naming_h(a_t+1, b_t, h_num[i_k], o_f, o_i, o_j, dist_layers)
                    + "_nu=" + to_string(inu);
                  string name_old = misc::naming_h(a_t, b_t, h_num[i_k], o_f, o_i, o_j, dist_layers)
                    + "_nu=" + to_string(inu);
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

