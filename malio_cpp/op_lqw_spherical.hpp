
#pragma once

#include <vector>
#include <map>
#include "Vector3.hpp"
#include "ml.hpp"
#include "SphericalHarmonics.hpp"

using namespace std;

struct LQW_SETTINGS
{
  int ave_times;
  vector<int> o_factor;
  vector<int> oi_oj;
  int b_in_Q;
  vector<int> l_in_Q;
  vector<string> p_in_Q;
};

class op_lqw_spherical
{
public:
  static map<string, vector<double>> spherical_order_parameter(const vector<Vector3>& coord,
                                                               const vector<Vector3>& direct,
                                                               const Vector3& box_length,
                                                               const LQW_SETTINGS& setting,
                                                               const vector<vector<int>>& neighbor_list);
  static map<string, vector<double>> w_order_parameter(const vector<Vector3>& coord,
                                                       const vector<Vector3>& direct,
                                                       const Vector3& box_length,
                                                       const LQW_SETTINGS& setting,
                                                       const vector<vector<int>>& neighbor_list);
  
private:
  static double func_to_value_wigner(int l_sph, const vector<vector<vector<double>>>& wigner3j,
                                     const vector<COMPLEX>& func); 
  static vector<vector<vector<double>>> gen_wigner3j(int l_sph);
  static double calc_q_norm(int l_sph, const vector<COMPLEX>& q_i_func);
  static double calc_qi_qj(const vector<vector<COMPLEX>>& q_func_name, int i_i,
                           const vector<vector<int>>& neighbor_list, int l_sph);
  static vector<COMPLEX> calc_qi_qj_func(const vector<vector<COMPLEX>>& q_func_name, int i_i,
                                         const vector<vector<int>>& neighbor_list, int l_sph);
  static COMPLEX sph_harm(int l_sph, int m_sph, double theta, double phi);
  static vector<COMPLEX> calc_q(int l_sph, double theta, double phi);
  static double func_to_value(int l_sph, const vector<COMPLEX>& func);
  static map<string, vector<COMPLEX>> calc_q_body(const vector<Vector3>& coord_ij,
                                                  const vector<Vector3>& direct_ij,
                                                  const vector<Vector3>& coord,
                                                  const vector<Vector3>& direct,
                                                  const Vector3& box_length,
                                                  const vector<int>& neighbor_list_ii,
                                                  int i_i,
                                                  const LQW_SETTINGS& setting);
  static map<string, vector<COMPLEX>> calc_q_wrapper(const vector<Vector3>& coord,
                                                     const vector<Vector3>& direct,
                                                     const Vector3& box_length,
                                                     const vector<int>& neighbor_list_ii,
                                                     int i_i,
                                                     const LQW_SETTINGS& setting);
  static map<string, vector<double>> calc_spherical_order_parameter(char calc_type,
                                                                    const vector<Vector3>& coord,
                                                                    const vector<Vector3>& direct,
                                                                    const Vector3& box_length,
                                                                    const LQW_SETTINGS& setting,
                                                                    const vector<vector<int>>& neighbor_list);
  

};

