
#pragma once

#include <vector>
#include <map>
#include "Vector3.hpp"
#include "ml.hpp"
#include "SphericalHarmonics.hpp"

using namespace std;

struct Q_SETTINGS
{
  int ave_times;
  vector<int> o_factor;
  vector<int> oi_oj;
  int b_in_Q;
  vector<int> l_in_Q;
  vector<string> p_in_Q;
};

class op_q_spherical
{
public:
  static map<string, vector<double>> spherical_order_parameter(const vector<Vector3>& coord,
                                                               const vector<Vector3>& direct,
                                                               const Vector3& box_length,
                                                               const Q_SETTINGS& settings,
                                                               const vector<vector<int>>& neighbor_list);
  static vector<COMPLEX> calc_q(int l_sph, double theta, double phi);
  
private:
  static map<string, vector<COMPLEX>> calc_q_wrapper(const vector<Vector3>& coord,
                                                    const vector<Vector3>& direct,
                                                    const Vector3& box_length,
                                                    const vector<int>& neighbor_list_ii,
                                                    int i_i,
                                                    const Q_SETTINGS& settings);
  static COMPLEX sph_harm(int l_sph, int m_sph, double theta, double phi);
  static double func_to_value(int l_sph, const vector<COMPLEX>& func);

};

