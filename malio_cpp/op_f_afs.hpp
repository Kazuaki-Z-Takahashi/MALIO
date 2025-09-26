
#pragma once

#include <vector>
#include <map>
#include "Vector3.hpp"
#include "ml.hpp"

using namespace std;

struct F_SETTINGS
{
  int ave_times;
  vector<int> o_factor;
  vector<int> oi_oj_ok;
  vector<double> l_list;
  vector<FUNC> func;
};

class op_f_afs
{
public:
  static map<string, vector<double>> afs_order_parameter(const vector<Vector3>& coord,
                                                         const vector<Vector3>& direct,
                                                         const Vector3& box_length,
                                                         const F_SETTINGS& settings,
                                                         const vector<vector<int>>& neighbor_list);
  static map<string, double> calc_f_wrapper(const vector<Vector3>& coord,
                                            const vector<Vector3>& direct,
                                            const Vector3& box_length,
                                            const vector<int>& neighbor_list_ii,
                                            int i_i,
                                            const F_SETTINGS& settings);
};

