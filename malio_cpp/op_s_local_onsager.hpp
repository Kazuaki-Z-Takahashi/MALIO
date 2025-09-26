
#pragma once

#include <vector>
#include <map>
#include "Vector3.hpp"

using namespace std;

struct S_SETTINGS
{
  int ave_times;
  vector<int> n_in_S;
};

class op_s_local_onsager
{
public:
  static map<string, vector<double>> onsager_order_parameter(const vector<Vector3>& coord,
                                                             const vector<Vector3>& direct,
                                                             const Vector3& box_length,
                                                             const S_SETTINGS& settings,
                                                             const vector<vector<int>>& neighbor_list);

private:
  static map<string, double> calc_s_wrapper(const vector<Vector3>& coord,
                                            const vector<Vector3>& direct,
                                            const Vector3& box_length,
                                            const vector<int>& neighbor_list_ii,
                                            int i_i,
                                            const S_SETTINGS& settings);
  static vector<double> calc_order_param(const vector<Vector3>& direct, int n_leg, const Vector3& ref_vec);

};

