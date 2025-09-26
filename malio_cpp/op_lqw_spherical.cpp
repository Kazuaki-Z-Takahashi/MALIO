
#include <algorithm>
#include <numeric>
#include "op_lqw_spherical.hpp"
#include "misc.hpp"
#include "sympy_physics.hpp"

double op_lqw_spherical::func_to_value_wigner(int l_sph, const vector<vector<vector<double>>>& wigner3j,
                                              const vector<COMPLEX>& func)
{
  int n = 2 * l_sph + 1;
  double sum_vec = 0.0;
  for( int m1 = -l_sph; m1 <= l_sph; ++m1) {
    for( int m2 = -l_sph; m2 <= l_sph; ++m2) {
      int m3 = -m1 - m2;
      if( -l_sph <= m3 && m3 <= l_sph ) {
        int i1 = (m1 + n) % n;
        int i2 = (m2 + n) % n;
        int i3 = (m3 + n) % n;
        double wig = wigner3j[i1][i2][i3];
        sum_vec += wig * (func[i1] * func[i2] * func[i3]).real();
      }
    }
  }

  double sum_norm2 = 0.0;
  for( int i_j = 0; i_j < 2 * l_sph + 1; ++i_j) {
    COMPLEX comp = func[i_j];
    sum_norm2 += (comp * conj(comp)).real();
  }
  double sum_norm = pow(sum_norm2, 3.0/2.0);

  double w_value = sum_vec / sum_norm;
  return round(w_value * 1e+14) * 1e-14;
}


vector<vector<vector<double>>> op_lqw_spherical::gen_wigner3j(int l_sph)
{
  int l2 = 2 * l_sph + 1;
  vector<vector<vector<double>>> wig(l2, vector<vector<double>>(l2, vector<double>(l2, 0.0)));
  for( int m1 = -l_sph; m1 <= l_sph; ++m1) {
    for( int m2 = -l_sph; m2 <= l_sph; ++m2) {
      int m3 = -m1 - m2;
      if( -l_sph <= m3 && m3 <= l_sph ) {
        int i1 = (m1 + l2) % l2;
        int i2 = (m2 + l2) % l2;
        int i3 = (m3 + l2) % l2;
        wig[i1][i2][i3] = sympy_physics::wigner_3j(l_sph, l_sph, l_sph, m1, m2, m3);
      }
    }
  }
  return wig;
}


double op_lqw_spherical::calc_q_norm(int l_sph, const vector<COMPLEX>& q_i_func)
{
  double norm2 = 0.0;
  for( int l = 0; l < 2 * l_sph + 1; ++l ) {
    norm2 += (q_i_func[l] * conj(q_i_func[l])).real();
  }
  return sqrt(norm2);
}


double op_lqw_spherical::calc_qi_qj(const vector<vector<COMPLEX>>& q_func_name, int i_i,
                                    const vector<vector<int>>& neighbor_list, int l_sph)
{
  double a_ij = 0.0;
  vector<COMPLEX> q_i_func(2 * l_sph + 1);
  for(int i = 0; i < 2 * l_sph + 1; ++i) {
    q_i_func[i] = q_func_name[i_i][i];
  }

  for( int i_j : neighbor_list[i_i] ) {
    vector<COMPLEX> q_j_func(2 * l_sph + 1);
    for(int i = 0; i < 2 * l_sph + 1; ++i) {
      q_j_func[i] = q_func_name[i_j][i];
    }
    double q_i_norm = calc_q_norm(l_sph, q_i_func);
    double q_j_norm = calc_q_norm(l_sph, q_j_func);

    double a_temp = 0.0;
    for( int l = 0; l < 2 * l_sph + 1; ++l) {
      a_temp += (q_i_func[l] * conj(q_j_func[l])).real() / (q_i_norm * q_j_norm);
    }
    a_ij += a_temp;
  }
  
  if( !neighbor_list[i_i].empty() ) {
    a_ij /= neighbor_list[i_i].size();
  }
  else {
    a_ij = 0.0;
  }

  return a_ij;
}

vector<COMPLEX> op_lqw_spherical::calc_qi_qj_func(const vector<vector<COMPLEX>>& q_func_name,int i_i,
                                                  const vector<vector<int>>& neighbor_list, int l_sph)
{
  vector<COMPLEX> q_i_func(2 * l_sph + 1);
  for(int i = 0; i < 2 * l_sph + 1; ++i) {
    q_i_func[i] = q_func_name[i_i][i];
  }

  vector<COMPLEX> q_ij_func(2 * l_sph + 1, 0.0);
  
  for( int i_j : neighbor_list[i_i] ) {
    vector<COMPLEX> q_j_func(2 * l_sph + 1);
    for(int i = 0; i < 2 * l_sph + 1; ++i) {
      q_j_func[i] = q_func_name[i_j][i];
    }
    double q_i_norm = calc_q_norm(l_sph, q_i_func);
    double q_j_norm = calc_q_norm(l_sph, q_j_func);

    for( int i = 0; i < 2 * l_sph + 1; ++i ) {
      q_ij_func[i] += (q_i_func[i] * conj(q_j_func[i])) / (q_i_norm * q_j_norm);
    }
  }
  
  if( !neighbor_list[i_i].empty() ) {
    for( int i = 0; i < 2 * l_sph + 1; ++i ) {
      q_ij_func[i] /= neighbor_list[i_i].size();
    }
  }
  else {
    q_ij_func = q_i_func;
  }
  
  return q_ij_func;
}


COMPLEX op_lqw_spherical::sph_harm(int l_sph, int m_sph, double theta, double phi)
{
  return SphericalHarmonics::Y(l_sph, m_sph, theta, phi);
}

vector<COMPLEX> op_lqw_spherical::calc_q(int l_sph, double theta, double phi)
{
  int n = 2 * l_sph + 1;
  vector<COMPLEX> q_l = SphericalHarmonics::Yn(l_sph, theta, phi);
  q_l.resize(n);
  for( int m_sph = -l_sph; m_sph < 0; ++m_sph) {
    q_l[m_sph + n] = q_l[abs(m_sph)];
  }
  
  return q_l;
}

double op_lqw_spherical::func_to_value(int l_sph, const vector<COMPLEX>& func)
{
  double sum_norm2 = 0.0;
  for( int i_j = 0; i_j < 2 * l_sph + 1; ++i_j) {
    COMPLEX comp = func[i_j];
    sum_norm2 += (comp * conj(comp)).real();
  }
  double q_value = sqrt(sum_norm2 * (4.0 * M_PI) / (2.0 * l_sph + 1.0));
  return round(q_value * 1e+14) * 1e-14;
}

map<string, vector<COMPLEX>>
op_lqw_spherical::calc_q_body(const vector<Vector3>& coord_ij,
                              const vector<Vector3>& direct_ij,
                              const vector<Vector3>& coord,
                              const vector<Vector3>& direct,
                              const Vector3& box_length,
                              const vector<int>& neighbor_list_ii,
                              int i_i,
                              const LQW_SETTINGS& setting)
{
  const Vector3& coord_ii  = coord [i_i];
  const Vector3& direct_ii = direct[i_i];
    
  int N = neighbor_list_ii.size();
              
  map<string, vector<COMPLEX>> q_func_temp;

  for( string p_weight : setting.p_in_Q ) {
    for( int o_j : setting.oi_oj ) {
      for( int o_i : setting.oi_oj ) {
        for( int o_f : setting.o_factor ) {
          for( int l_sph : setting.l_in_Q ) {
            string name = misc::naming_q(l_sph, 0, 0, o_f, o_i, o_j, p_weight);
            
            Vector3 x_i = misc::move_vec(coord_ii, direct_ii, o_f, o_i);

            // neighbor
            vector<COMPLEX> q_temp(2 * l_sph + 1, 0.0);
            for( int j = 0; j < (int)neighbor_list_ii.size(); ++j ) {
              Vector3 x_j = misc::move_vec(coord_ij[j], direct_ij[j], o_f, o_i);
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
            for( int i_j : setting.oi_oj )
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
              q_temp[i_k] /= (N + p_fact);
          
            q_func_temp[name] = q_temp;
          }
        }
      }
    }
  }

  return q_func_temp;
}

map<string, vector<COMPLEX>> op_lqw_spherical::calc_q_wrapper(const vector<Vector3>& coord,
                                                              const vector<Vector3>& direct,
                                                              const Vector3& box_length,
                                                              const vector<int>& neighbor_list_ii,
                                                              int i_i,
                                                              const LQW_SETTINGS& setting)
{
  vector<Vector3> coord_ij;
  vector<Vector3> direct_ij;
  for(int j = 0; j < (int)neighbor_list_ii.size(); ++j) {
    int i_j = neighbor_list_ii[j];
    coord_ij .push_back(coord [i_j]);
    direct_ij.push_back(direct[i_j]);
  }
  map<string, vector<COMPLEX>> q_func_temp = calc_q_body(coord_ij, direct_ij, coord, direct, box_length,
                                                         neighbor_list_ii, i_i, setting);
  return q_func_temp;
}


map<string, vector<double>> op_lqw_spherical::calc_spherical_order_parameter(char calc_type,
                                                                             const vector<Vector3>& coord,
                                                                             const vector<Vector3>& direct,
                                                                             const Vector3& box_length,
                                                                             const LQW_SETTINGS& setting,
                                                                             const vector<vector<int>>& neighbor_list)
{
  // [Q_N]_l_a_b_oi_oj_P
  vector<map<string, vector<COMPLEX>>> q_func_temp;

  int n_coord = coord.size();
  
  for(int i = 0; i < n_coord; ++i) {
    map<string, vector<COMPLEX>> q = calc_q_wrapper(coord, direct, box_length, neighbor_list[i], i, setting);
    q_func_temp.push_back(q);
  }

  map<string, vector<vector<COMPLEX>>> q_func = misc::data_num_name_to_data_name_num(q_func_temp, n_coord);

  // calc function average
  for( string p_weight : setting.p_in_Q ) {
    for( int o_j : setting.oi_oj ) {
      for( int o_i : setting.oi_oj ) {
        for( int o_f : setting.o_factor ) {
          for(int b_t = 0; b_t < setting.b_in_Q; ++b_t) {
            for( int l_sph : setting.l_in_Q ) {
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
  if( calc_type == 'Q' ) {

    for( int l_sph : setting.l_in_Q ) {
      for(int b_t = 0; b_t < setting.b_in_Q + 1; ++b_t) {
        for( string p_weight : setting.p_in_Q ) {
          for( int o_j : setting.oi_oj ) {
            for( int o_i : setting.oi_oj ) {
              for( int o_f : setting.o_factor ) {
                string name = misc::naming_q(l_sph, 0, b_t, o_f, o_i, o_j, p_weight);
                vector<double> q_val;
                for(int i_i = 0; i_i < n_coord; ++i_i) {
                  // calc spherical harmonics function i-j
                  double a_ij = calc_qi_qj(q_func[name], i_i, neighbor_list, l_sph);
                  q_val.push_back(a_ij);
                }
                op_data[name] = q_val;
              }
            }
          }
        }
      }
    }
    
  }
  else if( calc_type == 'W' ) {
    
    for( int l_sph : setting.l_in_Q ) {
      vector<vector<vector<double>>> wigner3j = gen_wigner3j(l_sph);
      for(int b_t = 0; b_t < setting.b_in_Q + 1; ++b_t) {
        for( string p_weight : setting.p_in_Q ) {
          for( int o_j : setting.oi_oj ) {
            for( int o_i : setting.oi_oj ) {
              for( int o_f : setting.o_factor ) {
                string name = misc::naming_w(l_sph, 0, b_t, o_f, o_i, o_j, p_weight);
                vector<double> w_val;
                for(int i_i = 0; i_i < n_coord; ++i_i) {
                  // calc spherical harmonics function i-j
                  vector<COMPLEX> q_ij_func = calc_qi_qj_func(q_func[name], i_i, neighbor_list, l_sph);
                  w_val.push_back(func_to_value_wigner(l_sph, wigner3j, q_ij_func));
                }
                op_data[name] = w_val;
              }
            }
          }
        }
      }
    }
    
  }
  
  
    // neighbor value averaging
  for( string p_weight : setting.p_in_Q ) {
    for( int o_j : setting.oi_oj ) {
      for( int o_i : setting.oi_oj ) {
        for( int o_f : setting.o_factor ) {
          for(int b_t = 0; b_t < setting.b_in_Q + 1; ++b_t) {
            for(int a_t = 0; a_t < setting.ave_times; ++a_t) {
              for( int l_sph : setting.l_in_Q ) {
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

map<string, vector<double>> op_lqw_spherical::spherical_order_parameter(const vector<Vector3>& coord,
                                                                        const vector<Vector3>& direct,
                                                                        const Vector3& box_length,
                                                                        const LQW_SETTINGS& setting,
                                                                        const vector<vector<int>>& neighbor_list)
{
  map<string, vector<double>> op_data = calc_spherical_order_parameter('Q', coord, direct, box_length, setting,
                                                                       neighbor_list);
  return op_data;
}

map<string, vector<double>> op_lqw_spherical::w_order_parameter(const vector<Vector3>& coord,
                                                                const vector<Vector3>& direct,
                                                                const Vector3& box_length,
                                                                const LQW_SETTINGS& setting,
                                                                const vector<vector<int>>& neighbor_list)
{
  map<string, vector<double>> op_data = calc_spherical_order_parameter('W', coord, direct, box_length, setting,
                                                                       neighbor_list);
  return op_data;
}


