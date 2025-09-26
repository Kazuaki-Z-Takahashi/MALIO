
#pragma once

#include <vector>
#include <map>
#include "Vector3.hpp"
#include "ml.hpp"

using namespace std;

struct I_SETTINGS
{
  int ave_times;
  vector<int> o_factor;
  vector<int> oi_oj_ok;
};

class op_i_top
{
public:
  static map<string, vector<double>> top_order_parameter(const vector<Vector3>& coord,
                                                         const vector<Vector3>& direct,
                                                         const Vector3& box_length,
                                                         const I_SETTINGS& settings,
                                                         const vector<vector<int>>& neighbor_list);
  static map<string, double> calc_i_wrapper(const vector<Vector3>& coord,
                                            const vector<Vector3>& direct,
                                            const Vector3& box_length,
                                            const vector<int>& neighbor_list_ii,
                                            int i_i,
                                            const I_SETTINGS& settings);
};

