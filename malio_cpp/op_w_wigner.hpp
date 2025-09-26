
#pragma once

#include <vector>
#include <map>
#include "Vector3.hpp"
#include "ml.hpp"
#include "SphericalHarmonics.hpp"

using namespace std;

struct W_SETTINGS
{
  int ave_times;
  vector<int> o_factor;
  vector<int> oi_oj;
  int b_in_Q;
  vector<int> l_in_Q;
  vector<string> p_in_Q;
};

class op_w_wigner
{
public:
  static map<string, vector<double>> w_order_parameter(const vector<Vector3>& coord,
                                                       const vector<Vector3>& direct,
                                                       const Vector3& box_length,
                                                       const W_SETTINGS& settings,
                                                       const vector<vector<int>>& neighbor_list);
private:
  static map<string, vector<COMPLEX>> calc_w_wrapper(const vector<Vector3>& coord,
                                                     const vector<Vector3>& direct,
                                                     const Vector3& box_length,
                                                     const vector<int>& neighbor_list_ii,
                                                     int i_i,
                                                     const W_SETTINGS& settings);
  static vector<vector<vector<double>>> gen_wigner3j(int l_sph);
  static double func_to_value(int l_sph, const vector<vector<vector<double>>>& wigner3j,
                              const vector<COMPLEX>& func);

};

