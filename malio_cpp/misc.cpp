
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <complex.h>
#include <pocketfft_hdronly.h>
#include "misc.hpp"
#include "SphericalHarmonics.hpp"

using namespace pocketfft;

Vector3 misc::q_to_xyz(const Quaternion& q)
{
  return q.Rotate(Vector3(1.0, 0.0, 0.0));
}

Vector3 misc::calc_delta(const Vector3& x_end, const Vector3& x_start, const Vector3& box_length)
{
  Vector3 delta = x_end - x_start;
  if( delta.x <  - box_length.x / 2.0 ) delta.x += box_length.x;
  if( delta.y <  - box_length.y / 2.0 ) delta.y += box_length.y;
  if( delta.z <  - box_length.z / 2.0 ) delta.z += box_length.z;
  if( delta.x >=   box_length.x / 2.0 ) delta.x -= box_length.x;
  if( delta.y >=   box_length.y / 2.0 ) delta.y -= box_length.y;
  if( delta.z >=   box_length.z / 2.0 ) delta.z -= box_length.z;
  return delta;
}

void misc::sort_by_distance(vector<vector<int>>& nei_list, vector<vector<double>>& nei_dist)
{
  for(int i = 0; i < (int)nei_list.size(); ++i) {
    sort_by_distance(nei_list[i], nei_dist[i]);
  }
}

void misc::sort_by_distance(vector<int>& nei_list, vector<double>& nei_dist)
{
  vector<double> temp_dist = nei_dist;
  vector<int   > temp_list = nei_list;

  int n = temp_dist.size();
    
  vector<size_t> idx(n);
  iota(idx.begin(), idx.end(), 0);
  
  sort(idx.begin(), idx.end(), [&temp_dist](size_t i1, size_t i2) {
    return temp_dist[i1] < temp_dist[i2];
  });
  
  for(int j = 0; j < n; ++j) {
    nei_list[j] = temp_list[idx[j]];
    nei_dist[j] = temp_dist[idx[j]]; 
  }
}

Vector3 misc::calc_head_coordinate(const vector<Vector3>& coord, const vector<Vector3>& direct,
                                   int i_i, int o_f, int o_i)
{
  Vector3 coord_ii  = coord [i_i];
  Vector3 direct_ii = direct[i_i];
  Vector3 x_i = coord_ii + o_f * o_i * direct_ii;
  return  x_i;
}

double misc::distance_ik_jk(const Vector3& x_i, const Vector3& x_j, const Vector3& box_length, const Vector3& x_k)
{
  Vector3 x_ik = calc_delta(x_k, x_i, box_length);
  Vector3 x_jk = calc_delta(x_k, x_j, box_length);
  double distance = x_ik.Norm() + x_jk.Norm();
  return distance;
}

vector<int> misc::gen_neighbor_ij(const vector<Vector3>& coord, const vector<Vector3>& direct,
                                  const Vector3& box_length, const vector<int>& neighbor_list_ii,
                                  const Vector3& x_i, int i_j, int o_f, int o_j, int o_k, int m_neighbor)
{
  Vector3 x_j = calc_head_coordinate(coord, direct, i_j, o_f, o_j);

  vector<int> i_j_nei;
  vector<double> i_j_dist;
  
  for( int i_k : neighbor_list_ii ) {
    if( i_j == i_k )
      continue;
    Vector3 x_k = calc_head_coordinate(coord, direct, i_k, o_f, o_k);
    double dist = distance_ik_jk(x_i, x_j, box_length, x_k);

    add_index_to_list(i_j_nei, i_j_dist, m_neighbor, dist, i_k);
  }

  vector<vector<int   >> _i_j_nei (1, i_j_nei);
  vector<vector<double>> _i_j_dist(1, i_j_dist);
  sort_by_distance(_i_j_nei, _i_j_dist);
  
  return _i_j_nei[0];
}

vector<vector<int>> misc::gen_neighbor_ijk(const vector<Vector3>& coord, const vector<Vector3>& direct,
                                           const Vector3& box_length, const vector<int>& neighbor_list_ii,
                                           const Vector3& x_i, int o_f, int o_j, int o_k, int max_m)
{
  vector<vector<int>> neighbor_ijk;
  for( int i_j :  neighbor_list_ii) {
    vector<int> i_j_nei = gen_neighbor_ij(coord, direct, box_length, neighbor_list_ii,
                                          x_i, i_j, o_f, o_j, o_k, max_m);
    neighbor_ijk.push_back(i_j_nei);
  }
  
  return neighbor_ijk;
}

void misc::add_index_to_list(vector<int>& nei_ij, vector<double>& dist_ij,
                             int size, double dist, int index)
{
  if( (int)nei_ij.size() < size ) {
    nei_ij.push_back(index);
    dist_ij.push_back(dist);
  }
  else {
    auto it = max_element(begin(dist_ij), end(dist_ij));
    if( *it > dist ) {
      size_t idx = distance(dist_ij.begin(), it);
      nei_ij[idx] = index;
      dist_ij[idx] = dist;
    }
  }
  return;
}

string misc::naming_a(int a_t, const string& op_type, int m_nei, int o_f, int o_i, int o_j, int o_k)
{
  ostringstream oss;
  oss << "a="     << a_t
      << "_type=" << op_type
      << "_m="    << m_nei
      << "_of="   << o_f
      << "_oi="   << o_i
      << "_oj="   << o_j
      << "_ok="   << o_k;
  return oss.str();
}

string misc::naming_b(int a_t, int m_fac, double phi, int n_pow, int o_f, int o_i, int o_j, int o_k)
{
  ostringstream oss;
  oss << "a="    << a_t 
      <<"_m="    << m_fac
      << "_phi=" << phi
      << "_n="   << n_pow
      << "_of="  << o_f
      << "_oi="  << o_i
      << "_oj="  << o_j
      << "_ok="  << o_k;
  return oss.str();
}

string misc::naming_c(int a_t, int o_f, int o_i, int o_j, int o_k)
{
  ostringstream oss;
  oss << "a="     << a_t
      << "_of="   << o_f
      << "_oi="   << o_i
      << "_oj="   << o_j
      << "_ok="   << o_k;
  return oss.str();
}

string misc::naming_d(int a_t, int o_f, int o_i, int o_j, int o_k, int f_1, int f_2, int f_3)
{
  ostringstream oss;
  oss << "a="     << a_t
      << "_of="   << o_f
      << "_oi="   << o_i
      << "_oj="   << o_j
      << "_ok="   << o_k
      << "_f1="   << f_1
      << "_f2="   << f_2
      << "_f3="   << f_3;
  return oss.str();
}

string misc::naming_f(int a_t, int o_f, int o_i, int o_j, int o_k, int f_1, int f_2, int l_nei)
{
  ostringstream oss;
  oss << "a="     << a_t
      << "_of="   << o_f
      << "_oi="   << o_i
      << "_oj="   << o_j
      << "_ok="   << o_k
      << "_f1="   << f_1
      << "_f2="   << f_2
      << "_l="    << l_nei;
  return oss.str();
}

string misc::naming_h(int a_t, int b_t, int ibin, int o_f, int o_i, int o_j, int o_k)
{
  ostringstream oss;
  oss << "a="     << a_t
      << "_b="    << b_t
      << "_bin="  << ibin
      << "_of="   << o_f
      << "_oi="   << o_i
      << "_oj="   << o_j
      << "_ok="   << o_k;
  return oss.str();
}

string misc::naming_q(int l_sph, int a_t, int b_t, int o_f, int o_i, int o_j, const string& p_weight)
{
  ostringstream oss;
  oss << "l="     << l_sph
      << "_a="    << a_t
      << "_b="    << b_t
      << "_of="   << o_f
      << "_oi="   << o_i
      << "_oj="   << o_j
      << "_p="    << p_weight;
  return oss.str();
}

string misc::naming_w(int l_sph, int a_t, int b_t, int o_f, int o_i, int o_j, const string& p_weight)
{
  ostringstream oss;
  oss << "l="     << l_sph
      << "_a="    << a_t
      << "_b="    << b_t
      << "_of="   << o_f
      << "_oi="   << o_i
      << "_oj="   << o_j
      << "_p="    << p_weight;
  return oss.str();
}

string misc::naming_q2(int l_sph, const string& f_1, int a_t, int b_t, int o_f, int o_i, int o_j)
{
  ostringstream oss;
  oss << "l="     << l_sph
      << "_f1="   << f_1
      << "_a="    << a_t
      << "_b="    << b_t
      << "_of="   << o_f
      << "_oi="   << o_i
      << "_oj="   << o_j;
  return oss.str();
}

string misc::naming_w2(int l_sph, const string& f_1, int a_t, int b_t, int o_f, int o_i, int o_j)
{
  ostringstream oss;
  oss << "l="     << l_sph
      << "_f1="   << f_1
      << "_a="    << a_t
      << "_b="    << b_t
      << "_of="   << o_f
      << "_oi="   << o_i
      << "_oj="   << o_j;
  return oss.str();
}

string misc::naming_s(int a_t, int n_leg)
{
  ostringstream oss;
  oss << "a="     << a_t
      << "_n="    << n_leg;
  return oss.str();
}

string misc::naming_t(int a_t, int o_f, int o_i, int o_j, double dist_layers, int n_leg)
{
  ostringstream oss;
  oss << fixed << setprecision(1);
  oss << "a="     << a_t
      << "_n="    << n_leg
      << "_of="   << o_f
      << "_oi="   << o_i
      << "_oj="   << o_j
      << "_z="    << dist_layers;
  return oss.str();
}


template <typename T>
map<string, vector<T>> misc::data_num_name_to_data_name_num(const vector<map<string, T>>& a_dict,
                                                            int num_part)
{
  // data[i_i][name] => data[name][i_i]
  map<string, vector<T>> b_dict;
  for( auto it : a_dict[0] ) {
    string name = it.first;
    vector<T> a_temp;
    for( int i_i = 0; i_i < num_part; ++i_i ) {
      map<string, T> m = a_dict[i_i];
      a_temp.push_back(m[name]);
    }
    b_dict[name] = a_temp;
  }
  return b_dict;
}
template map<string, vector<double>>
misc::data_num_name_to_data_name_num(const vector<map<string, double>>& a_dict, int num_part);
template map<string, vector<vector<double>>>
misc::data_num_name_to_data_name_num(const vector<map<string, vector<double>>>& a_dict, int num_part);
template map<string, vector<vector<COMPLEX>>>
misc::data_num_name_to_data_name_num(const vector<map<string, vector<COMPLEX>>>& a_dict, int num_part);

vector<double> misc::v_neighb_ave(const vector<vector<int>>& neighbor_list, const vector<double>& val)
{
  vector<double> val_ave;
  for( int i_i = 0; i_i < (int)neighbor_list.size(); ++i_i ) {
    double ave = val[i_i];
    for( int inei : neighbor_list[i_i] ) {
      ave += val[inei];
    }
    ave /= neighbor_list[i_i].size() + 1;
    val_ave.push_back(ave);
  }
  return val_ave;
}

template <typename T>
vector<vector<T>> misc::v_neighb_ave(const vector<vector<int>>& neighbor_list,
                                     const vector<vector<T>>& val)
{
  vector<vector<T>> val_ave; // [[1,2,3],[1,2,3], ... ]
  for( int i_i = 0; i_i < (int)neighbor_list.size(); ++i_i ) {
    vector<T> part(val[i_i].size(), 0.0);
    for( int i_j = 0; i_j < (int)val[i_i].size(); ++i_j ) {
      T ave = val[i_i][i_j];
      for( int inei : neighbor_list[i_i] ) {
        ave += val[inei][i_j];
      }
      ave /= neighbor_list[i_i].size() + 1;
      part[i_j] = ave;
    }
    val_ave.push_back(part);
  }
  return val_ave;
}
template vector<vector<double>>
misc::v_neighb_ave(const vector<vector<int>>& neighbor_list, const vector<vector<double>>& val);
template vector<vector<COMPLEX>>
misc::v_neighb_ave(const vector<vector<int>>& neighbor_list, const vector<vector<COMPLEX>>& val);

Vector3 misc::move_vec(const Vector3& coord, const Vector3& direct, double o_factor, double orient)
{
  Vector3 x_i = coord + o_factor * orient * direct;
  return x_i;
}

double misc::angle(const Vector3& v_1, const Vector3& v_2)
{
  return acos(v_1 *v_2 / (v_1.Norm() * v_2.Norm()));
}

int misc::search_opposite_j_particle(const vector<Vector3>& coord, const vector<Vector3>& direct,
                                     const vector<int>& neighbor_list_ii, const Vector3& x_i, int i_j,
                                     const Vector3& x_j, const Vector3& box_length, int o_f, int o_k)
{
  Vector3 x_i_j = calc_delta(x_j, x_i, box_length);
  Vector3 x_j_opposite = x_i - x_i_j;

  int nearest_i_k = 1000;
  double nearest_distnace = 10000000.0;
  for( int i_k : neighbor_list_ii ) {
    if( i_k == i_j )
      continue;
    Vector3 x_k = calc_head_coordinate(coord, direct, i_k, o_f, o_k);
    Vector3 x_j_o_k = calc_delta(x_k, x_j_opposite, box_length);
    double distance = x_j_o_k.Norm();
    if( distance <= nearest_distnace ) {
      nearest_distnace = distance;
      nearest_i_k = i_k;
    }
  }

  return nearest_i_k;
}

vector<double> misc::fft_power(const vector<double>& v)
{
  int n = v.size();

  shape_t shape{ (size_t)n };
  shape_t axes = { 0 };
  stride_t stride_in  = { sizeof(std::complex<double>) };
  stride_t stride_out = { sizeof(std::complex<double>) };

  complex<double> data[n];
  for(int i = 0; i < n; ++i) {
    data[i] = v[i];
  }
    
  vector<complex<double>> res(n);

  c2c(shape, stride_in, stride_out, axes, FORWARD, data, res.data(), 1.0);

  vector<double> g_power(n);
  for(int i = 0; i < n; ++i) {
    g_power[i] = norm(res[i]);
  }

  return g_power;
}


bool misc::write_csv(const vector<DataFrame>& dfs, const string& csvfile)
{
  if( dfs.empty() )
    return false;
  
  ofstream ofs(csvfile.c_str());
  if( ofs.fail() )
    return false;
  ofs << fixed << setprecision(16);

  const vector<string>& header = dfs[0].GetHeader();
  const vector<vector<double>>& op = dfs[0].GetOrderParameter();
  
  int n = header.size();
  int m = op.size();

  // header
  ofs << ",";
  for( int i = 0; i < n; ++i ) {
    ofs << header[i] << ",";
  }
  ofs << "label" << endl;

  // order parameters
  for(int k = 0; k < (int)dfs.size(); ++k) {
    const vector<vector<double>>& op = dfs[k].GetOrderParameter();
    const string& label = dfs[k].GetLabel();
    
    for(int j = 0; j < m; ++j) {
      ofs << j << ",";
      for( int i = 0; i < n; ++i ) {
        ofs << op[j][i] << ",";
      }
      ofs << label << endl;
    }
  }

  ofs.close();

  return true;
}

void misc::convert_to_theta_phi(const Vector3& xyz, double& theta, double& phi)
{
  double dist = xyz.Norm();
  theta = acos(xyz.z / dist);
  phi = atan2(xyz.y, xyz.x);
}

double misc::average(const vector<double>& v)
{
  return accumulate(v.begin(), v.end(), 0.0) / v.size();
}

vector<double> misc::legendre(int n)
{
  double d = 1.0 / pow(2.0, n);

  vector<double> fact(2 * n + 1);
  fact[0] = 1.0;
  for(int i = 1; i <= 2 * n; ++i) {
    fact[i] = fact[i-1] * i;
  }
  
  vector<double> vd(n + 1, 0.0);
  for(int k = 0; 2 * k <= n; ++k) {
    vd[2*k] = pow(-1, k) * fact[2*n-2*k] / ( fact[k] * fact[n-2*k] * fact[n-k] ) * d;
  }

  return vd;
}

vector<double> misc::gen_z_plane(const Vector3& point, const Vector3& direct)
{
  double a_v = direct.x;
  double b_v = direct.y;
  double c_v = direct.z;
  double d_v = -a_v * point.x - b_v * point.y - c_v * point.z;
  return vector<double>({a_v, b_v, c_v, d_v});
}

double misc::plane_point_distance(const vector<double>& plane_var, const Vector3& box_length, const Vector3& point)
{
  double a_v = plane_var[0];
  double b_v = plane_var[1];
  double c_v = plane_var[2];
  double d_v = plane_var[3];

  double distance[3] = {0, 0, 0};
  for(int i_i = -1; i_i <= 1; ++i_i) {
    Vector3 p_temp = point + i_i * box_length;
    double x_v = p_temp.x;
    double y_v = p_temp.y;
    double z_v = p_temp.z;
    distance[i_i + 1] = abs(a_v * x_v + b_v * y_v + c_v * z_v + d_v) /
      sqrt(a_v * a_v + b_v * b_v + c_v * c_v);
  }
  
  return min({distance[0], distance[1], distance[2]});
}
