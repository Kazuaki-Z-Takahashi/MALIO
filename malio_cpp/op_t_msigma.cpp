
#include <algorithm>
#include "op_t_msigma.hpp"
#include "misc.hpp"

map<string, double> op_t_msigma::calc_t_wrapper(const vector<Vector3>& coord,
                                                const vector<Vector3>& direct,
                                                const Vector3& box_length,
                                                const vector<int>& neighbor_list_ii,
                                                int i_i,
                                                const T_SETTINGS& settings)
{
  map<string, double> op_temp;
  
  for( int n_leg : settings.n_in_T ) {
    for( int o_j : settings.oi_oj ) {
      for( int o_i : settings.oi_oj ) {
        for( int o_f : settings.o_factor ) {
          const Vector3& direct_ii = direct[i_i];
          Vector3 x_i = misc::calc_head_coordinate(coord, direct, i_i, o_f, o_i);

          vector<double> plane_var;
          if( direct_ii.Norm() == 0.0 ) {
            plane_var =  misc::gen_z_plane(x_i, Vector3(0.0, 0.0, 1.0));
          }
          else {
            plane_var =  misc::gen_z_plane(x_i, direct_ii);
          }

          vector<double> sum_r(settings.d_in_T.size(), 0.0);
          for(int i_j : neighbor_list_ii) {
            const Vector3& direct_i_j = direct[i_j];
            Vector3 x_j = misc::calc_head_coordinate(coord, direct, i_j, o_f, o_i);

            double cos_theta = (direct_ii * direct_i_j) / (direct_ii.Norm() * direct_i_j.Norm());
            // legendre function
            vector<double> legend_fac = misc::legendre(n_leg);

            double s_part = 0.0;
            for(int i = 0; i < (int)legend_fac.size(); ++i) {
              // n = 2 : legend_fac = [1.5, 0.0, -0.5]
              s_part += legend_fac[i] * pow(cos_theta, n_leg - i);
            }

            double dist_from_plane = misc::plane_point_distance(plane_var, box_length, x_j);

            for(int i_k = 0; i_k < (int)settings.d_in_T.size(); ++i_k) {
              double dist_layers = settings.d_in_T[i_k];
              double cos_part = cos(2.0 * M_PI * dist_from_plane / dist_layers);
              sum_r[i_k] += cos_part * s_part;
            }
          }
          
          for(int i_k = 0; i_k < (int)settings.d_in_T.size(); ++i_k) {
            double dist_layers = settings.d_in_T[i_k];
            if( neighbor_list_ii.size() == 0 ) {
              sum_r[i_k] = 0.0;
            }
            else {
              sum_r[i_k] /= neighbor_list_ii.size();
            }

            string name = misc::naming_t(0, o_f, o_i, o_j, dist_layers, n_leg);
            op_temp[name] = sum_r[i_k];
          }
        }
      }
    }
  }

  return op_temp;
}


map<string, vector<double>> op_t_msigma::mcmillan_order_parameter(const vector<Vector3>& coord,
                                                                  const vector<Vector3>& direct,
                                                                  const Vector3& box_length,
                                                                  const T_SETTINGS& settings,
                                                                  const vector<vector<int>>& neighbor_list)
{
  vector<map<string, double>> op_val_temp;

  int n_coord = coord.size();
  
  for(int i = 0; i < n_coord; ++i) {
    map<string, double> op_temp_ = calc_t_wrapper(coord, direct, box_length, neighbor_list[i], i, settings);
    op_val_temp.push_back(op_temp_);
  }

  map<string, vector<double>> op_data = misc::data_num_name_to_data_name_num(op_val_temp, n_coord);

  for( int a_t = 0; a_t < settings.ave_times; ++a_t ) {
    for( int n_leg : settings.n_in_T ) {
      for( double dist_layers : settings.d_in_T ) {      
        for( int o_j : settings.oi_oj ) {
          for( int o_i : settings.oi_oj ) {
            for( int o_f : settings.o_factor ) {
              for(int a_t = 0; a_t < settings.ave_times; ++a_t) {
                string name = misc::naming_t(a_t + 1, o_f, o_i, o_j, dist_layers, n_leg);
                string name_old = misc::naming_t(a_t, o_f, o_i, o_j, dist_layers, n_leg);
                op_data[name] = misc::v_neighb_ave(neighbor_list, op_data[name_old]);
              }
            }
          }
        }
      }
    }
  }

  return op_data;
}

