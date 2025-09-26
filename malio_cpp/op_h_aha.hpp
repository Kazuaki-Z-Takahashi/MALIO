
#pragma once

#include <vector>
#include <map>
#include "Vector3.hpp"
#include "ml.hpp"

using namespace std;

struct H_SETTINGS
{
  int ave_times;
  vector<int> o_factor;
  vector<int> oi_oj_ok;
  int b_in_H;
  vector<int> hist_num;
  vector<int> nu;
};

class op_h_aha
{
public:
  static map<string, vector<double>> aha_order_parameter(const vector<Vector3>& coord,
                                                         const vector<Vector3>& direct,
                                                         const Vector3& box_length,
                                                         const H_SETTINGS& settings,
                                                         const vector<vector<int>>& neighbor_list);
  static map<string, vector<double>> calc_h_wrapper(const vector<Vector3>& coord,
                                                    const vector<Vector3>& direct,
                                                    const Vector3& box_length,
                                                    const vector<int>& neighbor_list_ii,
                                                    int i_i,
                                                    const H_SETTINGS& settings);
};

