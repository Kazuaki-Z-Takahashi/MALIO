
#pragma once

#include <vector>
#include <string>

using namespace std;

typedef double (*FUNC)(double);
typedef double (*V_FUNC)(int, const vector<double>&, const vector<double>&);

struct V_FUNC_NAME
{
public:
  V_FUNC_NAME(V_FUNC f, const string& s) : func(f), name(s) {}
  V_FUNC func;
  string name;
};

double f_1(double r);
double f_2(double r);
double f_3(double r);
double f_4(double r);
double f_5(double r);
double f1(int j, const vector<double>& voronoi_area_list, const vector<double>& distance_list);
double f2(int j, const vector<double>& voronoi_area_list, const vector<double>& distance_list);
double f3(int j, const vector<double>& voronoi_area_list, const vector<double>& distance_list);

struct OP_SETTINGS
{
  vector<int> neighbor;
  vector<double> radius;
  int ave_times;
  vector<int> oi_oj;
  vector<int> o_factor;
  vector<int> m_in_A;
  vector<string> op_types;
  vector<int> m_in_B;
  vector<int> n_in_B;
  vector<double> phi_in_B;
  vector<FUNC> function_in_D;
  vector<FUNC> function_in_F;
  vector<double> l_in_F;
  int b_in_H;
  vector<int> bin_in_H;
  vector<int> nu_in_H;
  vector<int> l_in_Q;
  int b_in_Q;
  vector<string> p_in_Q;
  vector<int> n_in_S;
  vector<int> n_in_T;
  vector<double> d_in_T;
  vector<V_FUNC_NAME> function_in_Q2;
  vector<string> Delaunay;
  vector<string> analysis_type;

public:
  OP_SETTINGS();
  void ReadJson(const string& json_file);
};

