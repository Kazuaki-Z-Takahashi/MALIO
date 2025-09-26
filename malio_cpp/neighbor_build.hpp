
#pragma once

#include <vector>
#include <map>
#include "Vector3.hpp"
#include "int3.hpp"
#include "ml.hpp"

class CellList1D
{
private:
  vector<vector<int>> v1d;

public:
  CellList1D(int nz): v1d(nz) {}
    
  vector<int>& operator[](int i) {
    return v1d[i];
  }

  const vector<int>& operator[](int i) const {
    return v1d[i];
  }
};

class CellList2D
{
private:
  vector<CellList1D> v2d;

public:
  CellList2D(int ny, int nz): v2d(ny, CellList1D(nz)) {}
    
  CellList1D& operator[](int i) {
    return v2d[i];
  }

  const CellList1D& operator[](int i) const {
    return v2d[i];
  }
};

class CellList3D
{
private:
  vector<CellList2D> v3d;

public:
  CellList3D() {}
  CellList3D(int nx, int ny, int nz): v3d(nx, CellList2D(ny, nz)) {}
    
  CellList2D& operator[](int i) {
    return v3d[i];
  }

  const CellList2D& operator[](int i) const {
    return v3d[i];
  }
};


class neighbor_build
{
public:
  class SmallRadiusError: public exception {
  };


public:
  static double calc_thresh(const Vector3& box_length, int n, int max_neighbor, double safe_factor);
  static void build_neighbor_list(vector<Vector3>& coord, const Vector3& box_length,
                                  const string& cond_mode, double cond_dist, int cond_num,
                                  vector<vector<int>>& nei_list, vector<vector<double>>& nei_dist);
  static void build_cell(vector<Vector3>& coord, const Vector3& box_length, double thresh_dist,
                         CellList3D& cell_list, Vector3& cell_length, int3& cell_size);
  static int3 coord_to_cell_num(const Vector3& coord, const Vector3& cell_length);
  static void add_num_dist(vector<int>& nei_list, vector<double>& nei_dist, int num, int i_j, double dist);
  static void wrapper_cell_calc(const vector<Vector3>& coord, int i_i, const Vector3& cell_length,
                                const int3& cell_size, const CellList3D& cell_list, const Vector3& box_length,
                                const string& cond_mode, double cond_dist, int cond_num,
                                vector<int>& nei_list, vector<double>& nei_dist);
  static vector<int3> build_neighbor_cell(const int3& cell, const int3& cell_size);
  static void mod_neighbor_list(const vector<vector<int>>& nei_list, const vector<vector<double>>& nei_dist,
                                int neighbor, double raddi,
                                vector<vector<int>>& new_list, vector<vector<double>>& new_dist);
  static void build_neighbor_wrapper(vector<Vector3>& coord, const Vector3& box_length,
                                     const OP_SETTINGS& op_settings,
                                     map<string, vector<vector<int   >>>& neighbors_list,
                                     map<string, vector<vector<double>>>& neighbors_dist,
                                     map<string, vector<vector<double>>>& neighbors_area);

private:
  static vector<Vector3> add_mirror_image(const vector<Vector3>& orig_coord, const Vector3& sim_box);
  static void build_neighbor_delaunay(const vector<Vector3>& orig_coord, const Vector3& sim_box,
                                      vector<vector<int   >>& nei_list,
                                      vector<vector<double>>& nei_dist,
                                      vector<vector<double>>& nei_area);

};


