
#include <algorithm>
#include "op_s_local_onsager.hpp"
#include "misc.hpp"

vector<double> op_s_local_onsager::calc_order_param(const vector<Vector3>& direct, int n_leg, const Vector3& ref_vec)
{
  // second Legendre polynomial or Onsarger's order parameter
  // direct = [ [0,1,1], [1,2,3], ... ]

  // legendre function
  vector<double> legend_fac = misc::legendre(n_leg);

  vector<double> order_param;
  for( Vector3 i_dir : direct ) {
    // length = np.sqrt(np.dot(x_coord,x_coord))
    i_dir.Normalize();
    double cos_theta = ref_vec * i_dir;

    double temp = 0.0;
    for( int i = 0; i < (int)legend_fac.size(); ++i ) {
      // n = 2 : legend_fac = [1.5, 0.0, -0.5]
      temp += legend_fac[i] * pow(cos_theta, n_leg - i);
    }

    order_param.push_back(temp);
  }
  
  return order_param;
}

map<string, double> op_s_local_onsager::calc_s_wrapper(const vector<Vector3>& coord,
                                                 const vector<Vector3>& direct,
                                                 const Vector3& box_length,
                                                 const vector<int>& neighbor_list_ii,
                                                 int i_i,
                                                 const S_SETTINGS& settings)
{
  const Vector3& direct_ii = direct[i_i];
  // order parameter
  vector<Vector3> part_direct;
  for( int i_j : neighbor_list_ii ) {
    const Vector3& direct_i_j = direct[i_j];
    part_direct.push_back(direct_i_j);
  }

  map<string, double> op_temp;
  for( int n_leg : settings.n_in_S ) {
    vector<double> order_param = calc_order_param(part_direct, n_leg, direct_ii);
    string name = misc::naming_s(0, n_leg);
    op_temp[name] = misc::average(order_param);
  }

  return op_temp;
}


map<string, vector<double>> op_s_local_onsager::onsager_order_parameter(const vector<Vector3>& coord,
                                                                  const vector<Vector3>& direct,
                                                                  const Vector3& box_length,
                                                                  const S_SETTINGS& settings,
                                                                  const vector<vector<int>>& neighbor_list)
{
  vector<map<string, double>> op_val_temp;

  int n_coord = coord.size();
  
  for(int i = 0; i < n_coord; ++i) {
    map<string, double> op_temp_ = calc_s_wrapper(coord, direct, box_length, neighbor_list[i], i, settings);
    op_val_temp.push_back(op_temp_);
  }

  map<string, vector<double>> op_data = misc::data_num_name_to_data_name_num(op_val_temp, n_coord);

  for( int a_t = 0; a_t < settings.ave_times; ++a_t ) {
    for( int n_leg : settings.n_in_S ) {
      string name = misc::naming_s(a_t + 1, n_leg);
      string name_old = misc::naming_s(a_t, n_leg);
      op_data[name] = misc::v_neighb_ave(neighbor_list, op_data[name_old]);
    }
  }

  return op_data;
}

