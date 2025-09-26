
#pragma once

#include <vector>
#include <map>
#include "Vector3.hpp"
#include "ml.hpp"

using namespace std;

struct D_SETTINGS
{
  int ave_times;
  vector<int> o_factor;
  vector<int> oi_oj_ok;
  vector<FUNC> func;
};

class op_d_nda
{
public:
  static map<string, vector<double>> nda_order_parameter(const vector<Vector3>& coord,
                                                         const vector<Vector3>& direct,
                                                         const Vector3& box_length,
                                                         const D_SETTINGS& settings,
                                                         const vector<vector<int>>& neighbor_list);
  static map<string, double> calc_d_wrapper(const vector<Vector3>& coord,
                                            const vector<Vector3>& direct,
                                            const Vector3& box_length,
                                            const vector<int>& neighbor_list_ii,
                                            int i_i,
                                            const D_SETTINGS& settings);
};

