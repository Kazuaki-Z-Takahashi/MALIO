
#pragma once

#include <vector>
#include <map>
#include "Vector3.hpp"

using namespace std;

struct T_SETTINGS
{
  int ave_times;
  vector<int> n_in_S;
  vector<int> oi_oj;
  vector<int> o_factor;
  vector<int> n_in_T;
  vector<double> d_in_T;
};

class op_t_msigma
{
public:
  static map<string, vector<double>> mcmillan_order_parameter(const vector<Vector3>& coord,
                                                              const vector<Vector3>& direct,
                                                              const Vector3& box_length,
                                                              const T_SETTINGS& settings,
                                                              const vector<vector<int>>& neighbor_list);

private:
  static map<string, double> calc_t_wrapper(const vector<Vector3>& coord,
                                            const vector<Vector3>& direct,
                                            const Vector3& box_length,
                                            const vector<int>& neighbor_list_ii,
                                            int i_i,
                                            const T_SETTINGS& settings);
};

