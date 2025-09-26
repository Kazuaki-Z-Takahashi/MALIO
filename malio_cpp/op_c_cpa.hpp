
#pragma once

#include <vector>
#include <map>
#include "Vector3.hpp"

using namespace std;

struct C_SETTINGS
{
  int ave_times;
  vector<int> o_factor;
  vector<int> oi_oj_ok;
};

class op_c_cpa
{
public:
  static map<string, vector<double>> cpa_order_parameter(const vector<Vector3>& coord,
                                                         const vector<Vector3>& direct,
                                                         const Vector3& box_length,
                                                         const C_SETTINGS& settings,
                                                         const vector<vector<int>>& neighbor_list);
  static map<string, double> calc_cpa_wrapper(const vector<Vector3>& coord,
                                              const vector<Vector3>& direct,
                                              const Vector3& box_length,
                                              const vector<int>& neighbor_list_ii,
                                              int i_i,
                                              const C_SETTINGS& settings);
};

