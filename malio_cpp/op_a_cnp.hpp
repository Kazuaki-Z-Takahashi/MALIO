
#pragma once

#include <vector>
#include <map>
#include "Vector3.hpp"

using namespace std;

struct A_SETTINGS
{
  int ave_times;
  vector<int> m_in_A;
  vector<int> o_factor;
  vector<int> oi_oj_ok;
  vector<string> op_types;
};

class op_a_cnp
{
public:
  static map<string, vector<double>> cnp_order_parameter(const vector<Vector3>& coord,
                                                         const vector<Vector3>& direct,
                                                         const Vector3& box_length,
                                                         const A_SETTINGS& settings,
                                                         const vector<vector<int>>& neighbor_list);
  static map<string, double> calc_cnp_wrapper(const vector<Vector3>& coord,
                                              const vector<Vector3>& direct,
                                              const Vector3& box_length,
                                              const vector<int>& neighbor_list_ii,
                                              int i_i,
                                              const A_SETTINGS& settings);
};

