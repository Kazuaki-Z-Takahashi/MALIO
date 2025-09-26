
#pragma once

#include <vector>
#include <map>
#include "Vector3.hpp"
#include "Quaternion.hpp"
#include "DataFrame.hpp"

using namespace std;

class misc
{
public:
  static Vector3 q_to_xyz(const Quaternion& q);
  static Vector3 calc_delta(const Vector3& x_end, const Vector3& x_start, const Vector3& box_length);
  static void sort_by_distance(vector<int>& nei_list, vector<double>& nei_dist);
  static void sort_by_distance(vector<vector<int>>& nei_list, vector<vector<double>>& nei_dist);
  static Vector3 calc_head_coordinate(const vector<Vector3>& coord, const vector<Vector3>& direct,
                                      int i_i, int o_f, int o_i);
  static double distance_ik_jk(const Vector3& x_i, const Vector3& x_j,
                               const Vector3& box_length, const Vector3& x_k);
  static vector<int> gen_neighbor_ij(const vector<Vector3>& coord, const vector<Vector3>& direct,
                                     const Vector3& box_length, const vector<int>& neighbor_list_ii,
                                     const Vector3& x_i, int i_j, int o_f, int o_j, int o_k, int m_neighbor);
  static vector<vector<int>> gen_neighbor_ijk(const vector<Vector3>& coord, const vector<Vector3>& direct,
                                              const Vector3& box_length, const vector<int>& neighbor_list_ii,
                                              const Vector3& x_i, int o_f, int o_j, int o_k, int max_m);
  static void add_index_to_list(vector<int>& nei_ij, vector<double>& dist_ij,
                                int size, double dist, int index);
  static string naming_a(int a_t, const string& op_type, int m_nei, int o_f, int o_i, int o_j, int o_k);
  static string naming_b(int a_t, int m_fac, double phi, int n_pow, int o_f, int o_i, int o_j, int o_k);
  static string naming_c(int a_t, int o_f, int o_i, int o_j, int o_k);
  static string naming_d(int a_t, int o_f, int o_i, int o_j, int o_k, int f_1, int f_2, int f_3);
  static string naming_f(int a_t, int o_f, int o_i, int o_j, int o_k, int f_1, int f_2, int l_nei);
  static string naming_h(int a_t, int b_t, int ibin, int o_f, int o_i, int o_j, int o_k);
  static string naming_q(int l_sph, int a_t, int b_t, int o_f, int o_i, int o_j, const string& p_weight);
  static string naming_w(int l_sph, int a_t, int b_t, int o_f, int o_i, int o_j, const string& p_weight);
  static string naming_q2(int l_sph, const string& f_1, int a_t, int b_t, int o_f, int o_i, int o_j);
  static string naming_w2(int l_sph, const string& f_1, int a_t, int b_t, int o_f, int o_i, int o_j);
  static string naming_s(int a_t, int n_leg);
  static string naming_t(int a_t, int o_f, int o_i, int o_j, double dist_layers, int n_leg);
  template <typename T>
  static map<string, vector<T>> data_num_name_to_data_name_num(const vector<map<string, T>>& a_dict, int num_part);
  static vector<double> v_neighb_ave(const vector<vector<int>>& neighbor_list, const vector<double>& val);
  template <typename T>
  static vector<vector<T>> v_neighb_ave(const vector<vector<int>>& neighbor_list, const vector<vector<T>>& val);
  static Vector3 move_vec(const Vector3& coord, const Vector3& direct, double o_factor, double orient);
  static double angle(const Vector3& v_1, const Vector3& v_2);
  static int search_opposite_j_particle(const vector<Vector3>& coord, const vector<Vector3>& direct,
                                        const vector<int>& neighbor_list_ii, const Vector3& x_i, int i_j,
                                        const Vector3& x_j, const Vector3& box_length, int o_f, int o_k);
  static vector<double> fft_power(const vector<double>& v);
  static bool write_csv(const vector<DataFrame>& dfs, const string& csvfile);
  static void convert_to_theta_phi(const Vector3& xyz, double& theta, double& phi);
  static double average(const vector<double>& v);
  static vector<double> legendre(int n_leg);
  static vector<double> gen_z_plane(const Vector3& point, const Vector3& direct);
  static double plane_point_distance(const vector<double>& plane_var, const Vector3& box_length,
                                     const Vector3& point);

};

