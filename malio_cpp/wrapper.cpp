
#include <algorithm>
#include <map>
#include <mpi.h>
#include "wrapper.hpp"
#include "misc.hpp"
#include "neighbor_build.hpp"
#include "op_a_cnp.hpp"
#include "op_b_baa.hpp"
#include "op_c_cpa.hpp"
#include "op_d_nda.hpp"
#include "op_f_afs.hpp"
#include "op_h_aha.hpp"
#include "op_i_top.hpp"
#include "op_q_spherical.hpp"
#include "op_w_wigner.hpp"
#include "op_s_local_onsager.hpp"
#include "op_t_msigma.hpp"
#include "op_z_user_define.hpp"
#include "op_lqw_spherical.hpp"
#include "op_qw1_spherical.hpp"
#include "op_qw2_spherical.hpp"

using namespace std;

DataFrame ml_lsa::op_analyze(const vector<Vector3>& coord,
                             const vector<Quaternion>& direct_q,
                             const Vector3& box_length,
                             const OP_SETTINGS op_settings)
{
  /*
    order parameter analyze
    :param method: method for analysis.  Please see the details in the manual.
    :param coord: = [[0,0,0],[1,0,0]]
    :param direct: = direction quaternion [[1,0,0,0], [1,0,0,0]] or [] for no direction vector particle
    :param box_length: = [10,10,10]
    :param op_settings: settings for calculating order parameters
    :param thread_num:
    :return op_data:
  */

  int n = coord.size();
  
  vector<Vector3> direct(n);
  for(int i = 0; i < n; ++i) {
    direct[i] = misc::q_to_xyz(direct_q[i]);
  }

  return op_analyze(coord, direct, box_length, op_settings);
}

DataFrame ml_lsa::op_analyze(const vector<Vector3>& _coord,
                             const vector<Vector3>& _direct,
                             const Vector3& box_length,
                             const OP_SETTINGS op_settings)
{
  /*
    order parameter analyze
    :param method: method for analysis.  Please see the details in the manual.
    :param coord: = [[0,0,0],[1,0,0]]
    :param direct: = direction vector [[1,0,0],[1,0,0]] for no direction vector particle
    :param box_length: = [10,10,10]
    :param op_settings: settings for calculating order parameters
    :param thread_num:
    :return op_data:
  */

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  vector<Vector3> coord = _coord;
  vector<Vector3> direct = _direct;
  int n = coord.size();
  for(int i = 0; i < n; ++i) {
    direct[i].Normalize();
  }

  // build neighbor list
  map<string, vector<vector<int   >>> neighbors_list;
  map<string, vector<vector<double>>> neighbors_dist;
  map<string, vector<vector<double>>> neighbors_area;
  neighbor_build::build_neighbor_wrapper(coord, box_length, op_settings,
                                         neighbors_list, neighbors_dist, neighbors_area);
  
  // analyze
  DataFrame df_all;
  for( auto it = neighbors_list.begin(); it != neighbors_list.end(); ++it ) {
    string NR_name = it->first;
    if( rank == 0 )
      cout << NR_name << endl;
    const vector<vector<int>>& n_list = neighbors_list[NR_name];
    vector<vector<double>> neighbor_area;
    if( NR_name == "Delaunay" ) {
      neighbor_area = neighbors_area[NR_name];
    }
    else {
      neighbor_area = vector<vector<double>>(n);
      for(int i = 0; i < n; ++i) {
        neighbor_area[i] = vector<double>(n_list[i].size(), 1.0);
      }
    }
      
    map<string, vector<double>> op_data = op_analyze_with_neighbor_list(coord, direct, box_length, NR_name,
                                                                        op_settings, n_list, neighbor_area);
    DataFrame df(op_data, "");
    df_all.AddColumns(df);
  }
  
  return df_all;
}

map<string, vector<double>> ml_lsa::op_analyze_with_neighbor_list(const vector<Vector3>& coord,
                                                                  const vector<Vector3>& direct,
                                                                  const Vector3& box_length,
                                                                  const string& NR_name,
                                                                  const OP_SETTINGS& op_settings,
                                                                  const vector<vector<int>>& n_list,
                                                                  const vector<vector<double>>& nei_area)
{
  /*
    Analyze structure
    :param method: method for analysis.
    Please see the details in readme file.
    :param coord: = [[0,0,0],[1,0,0]]
    :param direct: = [[1,0,0],[1,0,0]]
    :param box_length: = [10,10,10]
    :param NR_name: = 'N6'
    :param n_list: is neighbor list [[1],[0]]
    :param op_settings:
    :param thread_num:
    :return op_data: type(op_data) is dict.
  */

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  map<string, map<string, vector<double>>> op_temp;

  const vector<string>& v = op_settings.analysis_type;

  // common neighborhood parameter (CNP) A
  if (find(v.begin(), v.end(), "A") != v.end() ) {
    clock_t t_start = clock();
    
    A_SETTINGS setting;
    setting.ave_times = op_settings.ave_times;
    setting.m_in_A    = op_settings.m_in_A;
    setting.o_factor  = op_settings.o_factor;
    setting.oi_oj_ok  = op_settings.oi_oj;
    setting.op_types  = op_settings.op_types;

    op_temp["A_" + NR_name] = op_a_cnp::cnp_order_parameter(coord, direct, box_length, setting, n_list);

    clock_t t_end = clock();
    double elapsed = double(t_end - t_start) / CLOCKS_PER_SEC;
    if( rank == 0 )
      cout << "# CNP A elap time " << elapsed << endl;
  }

  // bond angle analysis (BAA) B
  if (find(v.begin(), v.end(), string("B")) != v.end() ) {
    clock_t t_start = clock();
    
    B_SETTINGS setting;
    setting.ave_times = op_settings.ave_times;
    setting.m         = op_settings.m_in_B;
    setting.phi       = op_settings.phi_in_B;
    setting.n         = op_settings.n_in_B;
    setting.o_factor  = op_settings.o_factor;
    setting.oi_oj_ok  = op_settings.oi_oj;

    op_temp["B_" + NR_name] = op_b_baa::baa_order_parameter(coord, direct, box_length, setting, n_list);

    clock_t t_end = clock();
    double elapsed = double(t_end - t_start) / CLOCKS_PER_SEC;
    if( rank == 0 )
      cout << "# BAA B elap time " << elapsed << endl;
  }

  // centrometry parameter analysis (CPA) C
  if (find(v.begin(), v.end(), string("C")) != v.end() ) {
    clock_t t_start = clock();
    
    C_SETTINGS setting;
    setting.ave_times = op_settings.ave_times;
    setting.o_factor  = op_settings.o_factor;
    setting.oi_oj_ok  = op_settings.oi_oj;

    op_temp["C_" + NR_name] = op_c_cpa::cpa_order_parameter(coord, direct, box_length, setting, n_list);

    clock_t t_end = clock();
    double elapsed = double(t_end - t_start) / CLOCKS_PER_SEC;
    if( rank == 0 )
      cout << "# CPA C elap time " << elapsed << endl;
  }

  // neighbor distance analysis (NDA) D
  if (find(v.begin(), v.end(), string("D")) != v.end() ) {
    clock_t t_start = clock();
    
    D_SETTINGS setting;
    setting.ave_times = op_settings.ave_times;
    setting.o_factor  = op_settings.o_factor;
    setting.oi_oj_ok  = op_settings.oi_oj;
    setting.func      = op_settings.function_in_D;

    op_temp["D_" + NR_name] = op_d_nda::nda_order_parameter(coord, direct, box_length, setting, n_list);

    clock_t t_end = clock();
    double elapsed = double(t_end - t_start) / CLOCKS_PER_SEC;
    if( rank == 0 )
      cout << "# NDA D elap time " << elapsed << endl;
  }

  // Angular Fourier Series like parameter (AFS) F
  if (find(v.begin(), v.end(), string("F")) != v.end() ) {
    clock_t t_start = clock();
    
    F_SETTINGS setting;
    setting.ave_times = op_settings.ave_times;
    setting.o_factor  = op_settings.o_factor;
    setting.oi_oj_ok  = op_settings.oi_oj;
    setting.func      = op_settings.function_in_F;
    setting.l_list    = op_settings.l_in_F;

    op_temp["F_" + NR_name] = op_f_afs::afs_order_parameter(coord, direct, box_length, setting, n_list);

    clock_t t_end = clock();
    double elapsed = double(t_end - t_start) / CLOCKS_PER_SEC;
    if( rank == 0 )
      cout << "# AFS F elap time " << elapsed << endl;
  }

  // angle histogram analysis (AHA) H
  if (find(v.begin(), v.end(), string("H")) != v.end() ) {
    clock_t t_start = clock();
    
    H_SETTINGS setting;
    setting.ave_times = op_settings.ave_times;
    setting.o_factor  = op_settings.o_factor;
    setting.oi_oj_ok  = op_settings.oi_oj;
    setting.b_in_H    = op_settings.b_in_H;
    setting.hist_num  = op_settings.bin_in_H;
    setting.nu        = op_settings.nu_in_H;

    op_temp["H_" + NR_name] = op_h_aha::aha_order_parameter(coord, direct, box_length, setting, n_list);

    clock_t t_end = clock();
    double elapsed = double(t_end - t_start) / CLOCKS_PER_SEC;
    if( rank == 0 )
      cout << "# AHA H elap time " << elapsed << endl;
  }

  // tetrahedral order parameter (TOP) I
  if (find(v.begin(), v.end(), string("I")) != v.end() ) {
    clock_t t_start = clock();
    
    I_SETTINGS setting;
    setting.ave_times = op_settings.ave_times;
    setting.o_factor  = op_settings.o_factor;
    setting.oi_oj_ok  = op_settings.oi_oj;

    op_temp["I_" + NR_name] = op_i_top::top_order_parameter(coord, direct, box_length, setting, n_list);

    clock_t t_end = clock();
    double elapsed = double(t_end - t_start) / CLOCKS_PER_SEC;
    if( rank == 0 )
      cout << "# TOP I elap time " << elapsed << endl;
  }

  // Spherical Order parameter Q
  if (find(v.begin(), v.end(), string("Q")) != v.end() ) {
    clock_t t_start = clock();
    
    Q_SETTINGS setting;
    setting.ave_times = op_settings.ave_times;
    setting.oi_oj     = op_settings.oi_oj;
    setting.o_factor  = op_settings.o_factor;
    setting.b_in_Q    = op_settings.b_in_Q;
    setting.l_in_Q    = op_settings.l_in_Q;
    setting.p_in_Q    = op_settings.p_in_Q;

    op_temp["Q_" + NR_name] = op_q_spherical::spherical_order_parameter(coord, direct, box_length, setting, n_list);

    clock_t t_end = clock();
    double elapsed = double(t_end - t_start) / CLOCKS_PER_SEC;
    if( rank == 0 )
      cout << "# Spherical Q elap time " << elapsed << endl;
  }
  
  // Wigner Order parameter W
  if (find(v.begin(), v.end(), string("W")) != v.end() ) {
    clock_t t_start = clock();
    
    W_SETTINGS setting;
    setting.ave_times = op_settings.ave_times;
    setting.oi_oj     = op_settings.oi_oj;
    setting.o_factor  = op_settings.o_factor;
    setting.b_in_Q    = op_settings.b_in_Q;
    setting.l_in_Q    = op_settings.l_in_Q;
    setting.p_in_Q    = op_settings.p_in_Q;

    op_temp["W_" + NR_name] = op_w_wigner::w_order_parameter(coord, direct, box_length, setting, n_list);

    clock_t t_end = clock();
    double elapsed = double(t_end - t_start) / CLOCKS_PER_SEC;
    if( rank == 0 )
      cout << "# Wigner W elap time " << elapsed << endl;
  }

  // Spherical Order parameter Q1 or Wigner Order parameter W1
  if ( (find(v.begin(), v.end(), string("Q1")) != v.end()) ||
       (find(v.begin(), v.end(), string("W1")) != v.end()) ) {
    clock_t t_start = clock();
    
    QW1_SETTINGS setting;
    setting.ave_times = op_settings.ave_times;
    setting.oi_oj     = op_settings.oi_oj;
    setting.o_factor  = op_settings.o_factor;
    setting.b_in_Q    = op_settings.b_in_Q;
    setting.l_in_Q    = op_settings.l_in_Q;
    setting.p_in_Q    = op_settings.p_in_Q;
    
    if ( (find(v.begin(), v.end(), string("Q1")) != v.end()) ) {
      op_temp["Q1_" + NR_name] = op_qw1_spherical::spherical_order_parameter(coord, direct, box_length, setting,
                                                                             n_list, nei_area);
      clock_t t_end = clock();
      double elapsed = double(t_end - t_start) / CLOCKS_PER_SEC;
      if( rank == 0 )
        cout << "# Spherical Q1 elap time " << elapsed << endl;
    }
    if ( (find(v.begin(), v.end(), string("W1")) != v.end()) ) {
      op_temp["W1_" + NR_name] = op_qw1_spherical::w_order_parameter(coord, direct, box_length, setting,
                                                                     n_list, nei_area);
      clock_t t_end = clock();
      double elapsed = double(t_end - t_start) / CLOCKS_PER_SEC;
      if( rank == 0 )
        cout << "# Wigner W1 elap time " << elapsed << endl;
    }
  }
  
  // Spherical Order parameter Q2 or Wigner Order parameter W2
  if ( (find(v.begin(), v.end(), string("Q2")) != v.end()) ||
       (find(v.begin(), v.end(), string("W2")) != v.end()) ) {
    clock_t t_start = clock();
    
    QW2_SETTINGS setting;
    setting.ave_times = op_settings.ave_times;
    setting.oi_oj     = op_settings.oi_oj;
    setting.o_factor  = op_settings.o_factor;
    setting.b_in_Q    = op_settings.b_in_Q;
    setting.l_in_Q    = op_settings.l_in_Q;
    setting.function_in_Q2 = op_settings.function_in_Q2;
    
    if ( (find(v.begin(), v.end(), string("Q2")) != v.end()) ) {
      op_temp["Q2_" + NR_name] = op_qw2_spherical::spherical_order_parameter(coord, direct, box_length, setting,
                                                                             n_list, nei_area);
      clock_t t_end = clock();
      double elapsed = double(t_end - t_start) / CLOCKS_PER_SEC;
      if( rank == 0 )
        cout << "# Spherical Q2 elap time " << elapsed << endl;
    }
    if ( (find(v.begin(), v.end(), string("W2")) != v.end()) ) {
      op_temp["W2_" + NR_name] = op_qw2_spherical::w_order_parameter(coord, direct, box_length, setting,
                                                                     n_list, nei_area);
      clock_t t_end = clock();
      double elapsed = double(t_end - t_start) / CLOCKS_PER_SEC;
      if( rank == 0 )
        cout << "# Wigner W2 elap time " << elapsed << endl;
    }
  }
  
  // Local Spherical Order parameter Q or Local Wigner Order parameter W2
  if ( (find(v.begin(), v.end(), string("LQ")) != v.end()) ||
       (find(v.begin(), v.end(), string("LW")) != v.end()) ) {
    clock_t t_start = clock();
    
    LQW_SETTINGS setting;
    setting.ave_times = op_settings.ave_times;
    setting.oi_oj     = op_settings.oi_oj;
    setting.o_factor  = op_settings.o_factor;
    setting.b_in_Q    = op_settings.b_in_Q;
    setting.l_in_Q    = op_settings.l_in_Q;
    setting.p_in_Q    = op_settings.p_in_Q;
    
    if ( (find(v.begin(), v.end(), string("LQ")) != v.end()) ) {
      op_temp["LQ_" + NR_name] = op_lqw_spherical::spherical_order_parameter(coord, direct, box_length, setting,
                                                                             n_list);
      clock_t t_end = clock();
      double elapsed = double(t_end - t_start) / CLOCKS_PER_SEC;
      if( rank == 0 )
        cout << "# Spherical LQ elap time " << elapsed << endl;
    }
    if ( (find(v.begin(), v.end(), string("LW")) != v.end()) ) {
      op_temp["LW_" + NR_name] = op_lqw_spherical::w_order_parameter(coord, direct, box_length, setting, n_list);
      clock_t t_end = clock();
      double elapsed = double(t_end - t_start) / CLOCKS_PER_SEC;
      if( rank == 0 )
        cout << "# Wigner LW elap time " << elapsed << endl;
    }
  }
  
  // Onsager Order parameter S
  if (find(v.begin(), v.end(), string("S")) != v.end() ) {
    clock_t t_start = clock();
    
    S_SETTINGS setting;
    setting.ave_times = op_settings.ave_times;
    setting.n_in_S    = op_settings.n_in_S;

    op_temp["S_" + NR_name] = op_s_local_onsager::onsager_order_parameter(coord, direct, box_length, setting,
                                                                          n_list);

    clock_t t_end = clock();
    double elapsed = double(t_end - t_start) / CLOCKS_PER_SEC;
    if( rank == 0 )
      cout << "# Onsager S elap time " << elapsed << endl;
  }

  // McMillan Order parameter T
  if (find(v.begin(), v.end(), string("T")) != v.end() ) {
    clock_t t_start = clock();
    
    T_SETTINGS setting;
    setting.ave_times = op_settings.ave_times;
    setting.oi_oj     = op_settings.oi_oj;
    setting.o_factor  = op_settings.o_factor;
    setting.n_in_T    = op_settings.n_in_T;
    setting.d_in_T    = op_settings.d_in_T;

    op_temp["T_" + NR_name] = op_t_msigma::mcmillan_order_parameter(coord, direct, box_length, setting, n_list);

    clock_t t_end = clock();
    double elapsed = double(t_end - t_start) / CLOCKS_PER_SEC;
    if( rank == 0 )
      cout << "# McMillian T elap time " << elapsed << endl;
  }
  
  // User define order parameter
  if (find(v.begin(), v.end(), string("Z")) != v.end() ) {
    clock_t t_start = clock();

    op_temp["Z_" + NR_name] = op_z_user_define::user_define_parameter(coord, direct, box_length, n_list);

    clock_t t_end = clock();
    double elapsed = double(t_end - t_start) / CLOCKS_PER_SEC;
    if( rank == 0 )
      cout << "# User define Z elap time " << elapsed << endl;
  }
  
  map<string, vector<double>> op_data;
  
  for (const auto& iname : op_temp) {
    for (const auto& jname : iname.second) {
      op_data[iname.first + "_" + jname.first] = jname.second;
    }
  }

  return op_data;
}

