
#pragma once

#include <vector>
#include <map>
#include "Vector3.hpp"

using namespace std;

struct B_SETTINGS
{
  int ave_times;
  vector<int> m;
  vector<int> n;
  vector<double> phi;
  vector<int> o_factor;
  vector<int> oi_oj_ok;
};

class op_b_baa
{
public:
  static map<string, vector<double>> baa_order_parameter(const vector<Vector3>& coord,
                                                         const vector<Vector3>& direct,
                                                         const Vector3& box_length,
                                                         const B_SETTINGS& settings,
                                                         const vector<vector<int>>& neighbor_list);
  static map<string, double> calc_baa_wrapper(const vector<Vector3>& coord,
                                              const vector<Vector3>& direct,
                                              const Vector3& box_length,
                                              const vector<int>& neighbor_list_ii,
                                              int i_i,
                                              const B_SETTINGS& settings);
};

