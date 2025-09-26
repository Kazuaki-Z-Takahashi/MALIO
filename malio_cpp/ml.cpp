
#include <numeric>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include "ml.hpp"

using namespace nlohmann;

//### Set local order parameters (LOPs) ###
double f_1(double r)
{
  return r;
}
double f_2(double r)
{
  return r * r;
}
double f_3(double r)
{
  return sqrt(r);
}
double f_4(double r)
{
  return 1.0 - exp(- pow(r - 0.3, 2) / (2 * pow(0.015, 2)));
}
double f_5(double r)
{
  return 0.5 + 0.5 * exp(-pow(r - 0.3, 2) / (2 * pow(0.015,2)));
}

//### voronoi_area functions
double f1(int j, const vector<double>& voronoi_area_list, const vector<double>& distance_list)
{
  double sum = reduce(begin(voronoi_area_list), end(voronoi_area_list));
  double weight = voronoi_area_list[j] / sum;
  return weight;
  return weight;
}

double f2(int j, const vector<double>& voronoi_area_list, const vector<double>& distance_list)
{
  double weight = 1.0 / distance_list.size();
  return weight;
}

double f3(int j, const vector<double>& voronoi_area_list, const vector<double>& distance_list)
{
  double sum_dist = 0;
  for(int i = 0; i < (int)distance_list.size(); ++i) {
    sum_dist += 1.0 / distance_list[i];
  }
  double weight = (1.0 / distance_list[j]) / sum_dist;
  return weight;
}

OP_SETTINGS::OP_SETTINGS() :
  // default values from wrapper.param_check_all
  ave_times(0),
  oi_oj    (1, 0),
  o_factor (1, 0),
  // A
  m_in_A   (1, 2),
  op_types (1, "A"),
  // B
  m_in_B   (1, 2),
  n_in_B   (1, 1),
  phi_in_B (1, 0),
  // C
  // D
  // H
  b_in_H   (0),
  bin_in_H (1, 12),
  nu_in_H  (1, 3),
  // I
  // Q
  // W
  l_in_Q   (1, 4),
  b_in_Q   (0),
  p_in_Q   (1, "0"),
  // S
  n_in_S   (1, 2),
  // T
  n_in_T   (1, 2)
{
  function_in_D  = vector<FUNC>{f_1, f_2, f_3, f_4, f_5};
  function_in_F  = vector<FUNC>{f_1, f_2, f_3, f_4, f_5};

  function_in_Q2 = vector<V_FUNC_NAME>{
    V_FUNC_NAME(f1, "f1"),
    V_FUNC_NAME(f2, "f2"), 
    V_FUNC_NAME(f3, "f3")};
};

void OP_SETTINGS::ReadJson(const string& json_file)
{
  ifstream ifs(json_file);
  if ( !ifs.is_open() ) {
    cerr << "ERROR: Could not open the setting file : " << json_file << endl;
    exit(1);
  }

  json j;
  ifs >> j;

  if( !j["neighbor"     ].is_null() ) neighbor      = j["neighbor"     ].get<vector<int>>();
  if( !j["radius"       ].is_null() ) radius        = j["radius"       ].get<vector<double>>();
  if( !j["ave_times"    ].is_null() ) ave_times     = j["ave_times"    ].get<int>();
  if( !j["oi_oj"        ].is_null() ) oi_oj         = j["oi_oj"        ].get<vector<int>>();
  if( !j["o_factor"     ].is_null() ) o_factor      = j["o_factor"     ].get<vector<int>>();
  if( !j["op_types"     ].is_null() ) op_types      = j["op_types"     ].get<vector<string>>();
  if( !j["m_in_A"       ].is_null() ) m_in_A        = j["m_in_A"       ].get<vector<int>>();
  if( !j["m_in_B"       ].is_null() ) m_in_B        = j["m_in_B"       ].get<vector<int>>();
  if( !j["n_in_B"       ].is_null() ) n_in_B        = j["n_in_B"       ].get<vector<int>>();
  if( !j["phi_in_B"     ].is_null() ) phi_in_B      = j["phi_in_B"     ].get<vector<double>>();
  if( !j["l_in_F"       ].is_null() ) l_in_F        = j["l_in_F"       ].get<vector<double>>();
  if( !j["b_in_H"       ].is_null() ) b_in_H        = j["b_in_H"       ].get<int>();
  if( !j["bin_in_H"     ].is_null() ) bin_in_H      = j["bin_in_H"     ].get<vector<int>>();
  if( !j["nu_in_H"      ].is_null() ) nu_in_H       = j["nu_in_H"      ].get<vector<int>>();
  if( !j["b_in_Q"       ].is_null() ) b_in_Q        = j["b_in_Q"       ].get<int>();
  if( !j["l_in_Q"       ].is_null() ) l_in_Q        = j["l_in_Q"       ].get<vector<int>>();
  if( !j["p_in_Q"       ].is_null() ) p_in_Q        = j["p_in_Q"       ].get<vector<string>>();
  if( !j["n_in_S"       ].is_null() ) n_in_S        = j["n_in_S"       ].get<vector<int>>();
  if( !j["n_in_T"       ].is_null() ) n_in_T        = j["n_in_T"       ].get<vector<int>>();
  if( !j["d_in_T"       ].is_null() ) d_in_T        = j["d_in_T"       ].get<vector<double>>();
  if( !j["analysis_type"].is_null() ) analysis_type = j["analysis_type"].get<vector<string>>();
  if( !j["Delaunay"     ].is_null() ) Delaunay      = j["Delaunay"     ].get<vector<string>>();

  vector<string> funcs;
  if( !j["func_in_D"].is_null() ) {
    function_in_D.clear();
    funcs = j["func_in_D"].get<vector<string>>();
    for(int i = 0; i < funcs.size(); ++i) {
      if( funcs[i] == "f_1" ) function_in_D.push_back(f_1);
      if( funcs[i] == "f_2" ) function_in_D.push_back(f_2);
      if( funcs[i] == "f_3" ) function_in_D.push_back(f_3);
      if( funcs[i] == "f_4" ) function_in_D.push_back(f_4);
      if( funcs[i] == "f_5" ) function_in_D.push_back(f_5);
    }
  }
  if( !j["func_in_F"].is_null() ) {
    function_in_F.clear();
    funcs = j["func_in_F"].get<vector<string>>();
    for(int i = 0; i < funcs.size(); ++i) {
      if( funcs[i] == "f_1" ) function_in_F.push_back(f_1);
      if( funcs[i] == "f_2" ) function_in_F.push_back(f_2);
      if( funcs[i] == "f_3" ) function_in_F.push_back(f_3);
      if( funcs[i] == "f_4" ) function_in_F.push_back(f_4);
      if( funcs[i] == "f_5" ) function_in_F.push_back(f_5);
    }
  }
  if( !j["func_in_Q2"].is_null() ) {
    function_in_Q2.clear();
    funcs = j["func_in_Q2"].get<vector<string>>();
    for(int i = 0; i < funcs.size(); ++i) {
      if( funcs[i] == "f1" ) function_in_Q2.push_back(V_FUNC_NAME(f1, "f1"));
      if( funcs[i] == "f2" ) function_in_Q2.push_back(V_FUNC_NAME(f2, "f2"));
      if( funcs[i] == "f3" ) function_in_Q2.push_back(V_FUNC_NAME(f3, "f3"));
    }
  }

  return;
}

