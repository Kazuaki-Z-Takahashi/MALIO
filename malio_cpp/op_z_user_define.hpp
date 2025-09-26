
#pragma once

#include <vector>
#include <map>
#include "Vector3.hpp"

using namespace std;

class op_z_user_define
{
public:
  static map<string, vector<double>> user_define_parameter(const vector<Vector3>& coord,
                                                           const vector<Vector3>& direct,
                                                           const Vector3& box_length,
                                                           const vector<vector<int>>& neighbor_list);
};

