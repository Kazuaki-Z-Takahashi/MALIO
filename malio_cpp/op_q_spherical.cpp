
#include <algorithm>
#include <numeric>
#include "op_q_spherical.hpp"
#include "misc.hpp"

COMPLEX op_q_spherical::sph_harm(int l_sph, int m_sph, double theta, double phi)
{
  return SphericalHarmonics::Y(l_sph, m_sph, theta, phi);
}

vector<COMPLEX> op_q_spherical::calc_q(int l_sph, double theta, double phi)
{
  int n = 2 * l_sph + 1;
  vector<COMPLEX> q_l = SphericalHarmonics::Yn(l_sph, theta, phi);
  q_l.resize(n);
  for( int m_sph = -l_sph; m_sph < 0; ++m_sph) {
    q_l[m_sph + n] = q_l[abs(m_sph)];
  }
  
  return q_l;
}

double op_q_spherical::func_to_value(int l_sph, const vector<COMPLEX>& func)
{
  double sum_norm2 = 0.0;
  for( int i_j = 0; i_j < 2 * l_sph + 1; ++i_j) {
    COMPLEX comp = func[i_j];
    sum_norm2 += (comp * conj(comp)).real();
  }
  double q_value = sqrt(sum_norm2 * (4.0 * M_PI) / (2.0 * l_sph + 1.0));
  return round(q_value * 1e+14) * 1e-14;
}

map<string, vector<COMPLEX>> op_q_spherical::calc_q_wrapper(const vector<Vector3>& coord,
                                                            const vector<Vector3>& direct,
                                                            const Vector3& box_length,
                                                            const vector<int>& neighbor_list_ii,
                                                            int i_i,
                                                            const Q_SETTINGS& settings)
{
  int N = neighbor_list_ii.size();
              
  map<string, vector<COMPLEX>> q_func_temp;

  for( string p_weight : settings.p_in_Q ) {
    for( int o_j : settings.oi_oj ) {
      for( int o_i : settings.oi_oj ) {
        for( int o_f : settings.o_factor ) {
          for( int l_sph : settings.l_in_Q ) {
            string name = misc::naming_q(l_sph, 0, 0, o_f, o_i, o_j, p_weight);
            
            Vector3 x_i = misc::calc_head_coordinate(coord, direct, i_i, o_f, o_i);

            // neighbor
            vector<COMPLEX> q_temp(2 * l_sph + 1, 0.0);
            for( int i_j : neighbor_list_ii ) {
              Vector3 x_j = misc::calc_head_coordinate(coord, direct, i_j, o_f, o_j);
              Vector3 delta = misc::calc_delta(x_i, x_j, box_length);
              double theta = 0.0;
              double phi = 0.0;
              misc::convert_to_theta_phi(delta, theta, phi);
              vector<COMPLEX> q_l = calc_q(l_sph, theta, phi);

              for( int i_k = 0; i_k < 2 * l_sph + 1; ++i_k)
                q_temp[i_k] += q_l[i_k];
            }
                      
            // self director
            double p_fact = 1.0;
            // p_weight == [ 'N', 'N/2', '2*N' ]
            if( p_weight == "N" ) {
              p_fact = N;
            }
            else if( p_weight == "N/2" ) {
              p_fact = N * 0.5;
            }
            else if( p_weight == "2*N" ) {
              p_fact = N * 2.0;
            }
            else {
              p_fact = stod(p_weight);
            }

            vector<int> oi_list_not_oi;
            for( int i_j : settings.oi_oj )
              if( i_j != o_i )
                oi_list_not_oi.push_back(i_j);
            for( int i_j : oi_list_not_oi ) {
              Vector3 x_j = coord[i_j] + direct[i_j];
              Vector3 delta = misc::calc_delta(x_i, x_j, box_length);
              double theta = 0.0;
              double phi = 0.0;
              misc::convert_to_theta_phi(delta, theta, phi);
              vector<COMPLEX> q_l = calc_q(l_sph, theta, phi);
              
              for( int i_k = 0; i_k < 2 * l_sph + 1; ++i_k)
                q_temp[i_k] += p_fact * q_l[i_k];
              }
            
            for( int i_k = 0; i_k < 2 * l_sph + 1; ++i_k)
              q_temp[i_k] = q_temp[i_k] / (N + p_fact);
          
            q_func_temp[name] = q_temp;
          }
        }
      }
    }
  }

  return q_func_temp;
}


map<string, vector<double>> op_q_spherical::spherical_order_parameter(const vector<Vector3>& coord,
                                                                      const vector<Vector3>& direct,
                                                                      const Vector3& box_length,
                                                                      const Q_SETTINGS& settings,
                                                                      const vector<vector<int>>& neighbor_list)
{
  // [Q_N]_l_a_b_oi_oj_P
  vector<map<string, vector<COMPLEX>>> q_func_temp;

  int n_coord = coord.size();
  
  for(int i = 0; i < n_coord; ++i) {
    map<string, vector<COMPLEX>> q = calc_q_wrapper(coord, direct, box_length, neighbor_list[i], i, settings);
    q_func_temp.push_back(q);
  }

  map<string, vector<vector<COMPLEX>>> q_func = misc::data_num_name_to_data_name_num(q_func_temp, n_coord);

  // calc function average
  for( string p_weight : settings.p_in_Q ) {
    for( int o_j : settings.oi_oj ) {
      for( int o_i : settings.oi_oj ) {
        for( int o_f : settings.o_factor ) {
          for(int b_t = 0; b_t < settings.b_in_Q; ++b_t) {
            for( int l_sph : settings.l_in_Q ) {
              string name = misc::naming_q(l_sph, 0, b_t + 1, o_f, o_i, o_j, p_weight);
              string name_old = misc::naming_q(l_sph, 0, b_t, o_f, o_i, o_j, p_weight);
              q_func[name] = misc::v_neighb_ave(neighbor_list, q_func[name_old]);
            }
          }
        }
      }
    }
  }

  // func to value
  map<string, vector<double>> op_data;
  for( int l_sph : settings.l_in_Q ) {
    for(int b_t = 0; b_t < settings.b_in_Q + 1; ++b_t) {
      for( string p_weight : settings.p_in_Q ) {
        for( int o_j : settings.oi_oj ) {
          for( int o_i : settings.oi_oj ) {
            for( int o_f : settings.o_factor ) {
              string name = misc::naming_q(l_sph, 0, b_t, o_f, o_i, o_j, p_weight);
              vector<double> q_val;
              for(int i_i = 0; i_i < n_coord; ++i_i) {
                double d = func_to_value(l_sph, q_func[name][i_i]);
                q_val.push_back(d);
              }
              op_data[name] = q_val;
            }
          }
        }
      }
    }
  }

  // neighbor value averaging
  for( string p_weight : settings.p_in_Q ) {
    for( int o_j : settings.oi_oj ) {
      for( int o_i : settings.oi_oj ) {
        for( int o_f : settings.o_factor ) {
          for(int b_t = 0; b_t < settings.b_in_Q + 1; ++b_t) {
            for(int a_t = 0; a_t < settings.ave_times; ++a_t) {
              for( int l_sph : settings.l_in_Q ) {
                string name = misc::naming_q(l_sph, a_t + 1, b_t, o_f, o_i, o_j, p_weight);
                string name_old = misc::naming_q(l_sph, a_t, b_t, o_f, o_i, o_j, p_weight);
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

