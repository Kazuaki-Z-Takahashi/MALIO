
#include <math.h>
#include <time.h>
#include <algorithm>
#include "neighbor_build.hpp"
#include "misc.hpp"
#include <voro++.hh>

using namespace voro;
using namespace std;

double neighbor_build::calc_thresh(const Vector3& box_length, int part_num, int target_num, double safe_factor)
{
  double density = double(part_num) / (box_length.x * box_length.y * box_length.z);
  double target_r = pow((target_num / (density * (4.0 / 3.0) * M_PI)),  (1.0 / 3.0));

  return safe_factor * target_r;
}

void neighbor_build::build_neighbor_list(vector<Vector3>& coord, const Vector3& box_length,
                                         const string& cond_mode, double cond_dist, int cond_num,
                                         vector<vector<int>>& nei_list, vector<vector<double>>& nei_dist)
{
  /* building neighbor list
     :param coord: = [[0,0,0],[1,0,0]]
     :param box_length: = [10,10,10]
     :param condition: =
     {'mode' : 'thresh' or 'neighbor',
               'neighbor' is number of neighborhood particles,
      'dist' : is radii of neighborhood particles. }
     :return [neighbor_list, neidhbor_distance]: is new neighbor list
  */

  clock_t t_start = clock();

  int n_coord = coord.size();
  nei_list = vector<vector<int   >>(n_coord, vector<int   >());
  nei_dist = vector<vector<double>>(n_coord, vector<double>());
    
  CellList3D cell_list;
  Vector3 cell_length;
  int3 cell_size;
  build_cell(coord, box_length, cond_dist, cell_list, cell_length, cell_size);

  if( cell_size[0] < 3 || cell_size[1] < 3 || cell_size[2] < 3 ) {
    for(int i_i = 0; i_i < n_coord - 1; ++i_i) {
      for(int i_j = i_i + 1; i_j < n_coord; ++i_j) {
        Vector3 delta = misc::calc_delta(coord[i_i], coord[i_j], box_length);
        double dist = delta.Norm();
        if( cond_mode == "thresh" ) {
          if( dist < cond_dist ) {
            nei_list[i_i].push_back(i_j);
            nei_dist[i_i].push_back(dist);
            nei_list[i_j].push_back(i_i);
            nei_dist[i_j].push_back(dist);
          }
        }
        else if( cond_mode == "neighbor" ) {
          add_num_dist(nei_list[i_i], nei_dist[i_i], cond_num, i_j, dist);
          add_num_dist(nei_list[i_j], nei_dist[i_j], cond_num, i_i, dist);
        }
      }
    }
    
    misc::sort_by_distance(nei_list, nei_dist);
    
    return;
  }

  for(int i = 0; i < n_coord; ++i) {
    wrapper_cell_calc(coord, i, cell_length, cell_size, cell_list, box_length, cond_mode, cond_dist, cond_num,
                      nei_list[i], nei_dist[i]);

  }

  for(int i = 0; i < (int)nei_list.size(); ++i) {
    for(int j = 0; j < (int)nei_list[i].size(); ++j) {
      int now_j = nei_list[i][j];
      if( i < now_j ) {
        if( cond_mode == "thresh" ) {
          nei_list[now_j].push_back(i);
          nei_dist[now_j].push_back(nei_dist[i][j]);
        }
        else if( cond_mode == "neighbor" ) {
          add_num_dist(nei_list[now_j], nei_dist[now_j], cond_num, i, nei_dist[i][j]);
        }
      }
    }
  }

  misc::sort_by_distance(nei_list, nei_dist);

  // check
  if( cond_mode == "neighbor" ) {
    if( (int)nei_list.size() < cond_num ) {
      cerr << "# neighbor num too big. you require " << cond_num
           << " neighbors. But there are " << nei_list.size() << "particles." << endl;
      exit(1);
      for(int i = 0; i < (int)nei_list.size(); ++i) {
        if( (int)nei_list[i].size() < cond_num ) {
          cerr << "# radius too small. you require " << cond_num
               << " neighbors. But there are " << nei_list[i].size() << "neighbors." << endl;
          exit(1);
        }
      }
    }
  }
  
  clock_t t_end = clock();
  double time = double(t_end - t_start) / CLOCKS_PER_SEC;
  cout << "# neighbor elap time " << time << endl;

  return;
}

void neighbor_build::build_cell(vector<Vector3>& coord, const Vector3& box_length, double thresh_dist,
                                CellList3D& cell_list, Vector3& cell_length, int3& cell_size)
{
  for(int i = 0; i < 3; ++i) {
    cell_size[i] = int(box_length[i] / thresh_dist);
    cell_length[i] = box_length[i] / double(cell_size[i]);
  }

  cell_list = CellList3D(cell_size[0], cell_size[1], cell_size[2]);

  for(int i = 0; i < (int)coord.size(); ++i) {
    // periodic boundary condition check
    Vector3 v = coord[i];
    while( v.x < 0.0          ) v.x += box_length.x;
    while( v.y < 0.0          ) v.y += box_length.y;
    while( v.z < 0.0          ) v.z += box_length.z;
    while( v.x > box_length.x ) v.x -= box_length.x;
    while( v.y > box_length.y ) v.y -= box_length.y;
    while( v.z > box_length.z ) v.z -= box_length.z;
    coord[i] = v;

    int3 inum = coord_to_cell_num(v, cell_length);
    cell_list[inum.i1][inum.i2][inum.i3].push_back(i);
  }
}

int3 neighbor_build::coord_to_cell_num(const Vector3& coord, const Vector3& cell_length)
{
  int3 inum;
  inum.i1 = int(coord.x / cell_length[0]);
  inum.i2 = int(coord.y / cell_length[1]);
  inum.i3 = int(coord.z / cell_length[2]);
  return inum;
}

void neighbor_build::add_num_dist(vector<int>& nei_list, vector<double>& nei_dist, int num, int i_j, double dist)
{
  // add i_j to nei_list and nei_dist
  if( (int)nei_list.size() < num ) {
    nei_list.push_back(i_j);
    nei_dist.push_back(dist);
  }
  else {
    auto it = max_element(nei_dist.begin(), nei_dist.end());
    if( *it > dist ) {
      size_t idx = distance(nei_dist.begin(), it);
      nei_list[idx] = i_j;
      nei_dist[idx] = dist;
    }
  }
}


vector<int3> neighbor_build::build_neighbor_cell(const int3& cell, const int3& cell_size)
{
  // input  : [0,0,0]
  // output : if cell_size == [4,4,4] , [-1,0,0] => [3,0,0], [4,0,0] =>
  // [0,0,0]

  vector<int3> neighbor;
    
  for(int iz = -1; iz <= 1; ++iz) {
    for(int iy = -1; iy <= 1; ++iy) {
      for(int ix = -1; ix <= 1; ++ix) {
        int3 ii = cell + int3(ix, iy, iz);
        if( ii.i1 == -1           ) ii.i1 = cell_size.i1 - 1;
        if( ii.i2 == -1           ) ii.i2 = cell_size.i2 - 1;
        if( ii.i3 == -1           ) ii.i3 = cell_size.i3 - 1;
        if( ii.i1 == cell_size.i1 ) ii.i1 = 0;
        if( ii.i2 == cell_size.i2 ) ii.i2 = 0;
        if( ii.i3 == cell_size.i3 ) ii.i3 = 0;
        neighbor.push_back(ii);
      }
    }
  }

  return neighbor;
}



void neighbor_build::wrapper_cell_calc(const vector<Vector3>& coord, int i_i, const Vector3& cell_length,
                                       const int3& cell_size, const CellList3D& cell_list, const Vector3& box_length,
                                       const string& cond_mode, double cond_dist, int cond_num,
                                       vector<int>& nei_list, vector<double>& nei_dist)
{
  nei_list.clear();
  nei_dist.clear();

  const Vector3& coord_ii = coord[i_i];
  
  int3 cell = coord_to_cell_num(coord_ii, cell_length);
  
  vector<int3> neighbor = build_neighbor_cell(cell, cell_size);

  for(int i = 0; i < (int)neighbor.size(); ++i) {
    const int3& inei = neighbor[i];

    for(int j = 0; j < (int)cell_list[inei.i1][inei.i2][inei.i3].size(); ++j) {

      int i_j = cell_list[inei.i1][inei.i2][inei.i3][j];
      
      if( i_i >= i_j ) continue;
      
      const Vector3& coord_i_j = coord[i_j];
      Vector3 delta = misc::calc_delta(coord_ii, coord_i_j, box_length);
      double dist = delta.Norm();
      
      if( cond_mode == "thresh" ) {
        if( dist <= cond_dist ) {
          nei_list.push_back(i_j);
          nei_dist.push_back(dist);
        }
        else if( cond_mode == "neighbor" ) {
          if( (int)nei_list.size() < cond_num ) {
            nei_list.push_back(i_j);
            nei_dist.push_back(dist);
          }
          else {
            auto it = max_element(nei_dist.begin(), nei_dist.end());
            if( *it > dist ) {
              size_t idx = distance(nei_dist.begin(), it);
              nei_list[idx] = i_j;
              nei_dist[idx] = dist;
            }
          }
        }
      }
      
    }
  }

  return;
}

void neighbor_build::mod_neighbor_list(const vector<vector<int>>& nei_list, const vector<vector<double>>& nei_dist,
                                       int neighbor, double radii,
                                       vector<vector<int>>& new_list, vector<vector<double>>& new_dist)
{
  /*
    cutting up neighbor_list
    Either A or B must be 0
    :param nei_list: is [[1,2],[0,2],[0,1]
    :param nei_dist: is [[1,2],[1,2],[1,2]]
    :param neighbor: is target number o_f neighbor atoms
    :param radii: is target distance o_f neighbor atoms
    :return [neighbor_list, neighbor_distance]: is cut up neighbor_list
  */

  int n_coord = nei_list.size();
  new_list = vector<vector<int   >>(n_coord, vector<int   >());
  new_dist = vector<vector<double>>(n_coord, vector<double>());

  if( (neighbor != 0) && (radii != 0.0) ) {
    cerr << "# error in mod neighbor list" << endl;
    return;
  }
  
  if( neighbor != 0 ) {
    for(int i = 0; i < (int)nei_list.size(); ++i) {
      if( (int)nei_list[i].size() < neighbor ) {
        cerr << "# radius too small. you require " << neighbor
             << " neighbors. But there are " << nei_list[i].size() << "neighbors." << endl;
        throw SmallRadiusError();
      }
      for(int j = 0; j < neighbor; ++j) {
        new_list[i].push_back(nei_list[i][j]);
        new_dist[i].push_back(nei_dist[i][j]);
      }
    }
  }
  else if( radii != 0.0 ) {
    for(int i = 0; i < (int)nei_list.size(); ++i) {
      for(int j = 0; j < (int)nei_list[i].size(); ++j) {
        double dist = nei_dist[i][j];
        if( dist <= radii ) {
          new_list[i].push_back(nei_list[i][j]);
          new_dist[i].push_back(nei_dist[i][j]);
        }
      }
    }
  }

  return;
}

vector<Vector3> neighbor_build::add_mirror_image(const vector<Vector3>& orig_coord, const Vector3& sim_box)
{
  vector<Vector3> coord = orig_coord;

  for(int iz = -1; iz <= 1; ++iz) {
    for(int iy = -1; iy <= 1; ++iy) {
      for(int ix = -1; ix <= 1; ++ix) {
        if(ix == 0 && iy == 0 && iz == 0)
          continue;
        for(int i = 0; i < (int)orig_coord.size(); ++i) {
          Vector3 v = orig_coord[i];
          v.x += ix * sim_box.x;
          v.y += iy * sim_box.y;
          v.z += iz * sim_box.z;
          coord.push_back(v);
        }
      }
    }
  }

  return coord;
}

void neighbor_build::build_neighbor_delaunay(const vector<Vector3>& orig_coord, const Vector3& sim_box,
                                             vector<vector<int   >>& neighbor_list,
                                             vector<vector<double>>& neighbor_dist,
                                             vector<vector<double>>& nei_area)
{ 
  neighbor_list.clear();
  neighbor_dist.clear();
  nei_area.clear();
  
  int n_part = orig_coord.size();
  if( n_part == 0 )
    return;

  vector<Vector3> coord = add_mirror_image(orig_coord, sim_box);

  const int n_x = 6, n_y = 6, n_z = 6;
  Vector3 vmin = coord[0];
  Vector3 vmax = coord[0];
  for(int i = 1; i < (int)coord.size(); ++i) {
    const Vector3& v = coord[i];
    vmin.x = min(vmin.x, v.x);
    vmin.y = min(vmin.y, v.y);
    vmin.z = min(vmin.z, v.z);
    vmax.x = max(vmax.x, v.x);
    vmax.y = max(vmax.y, v.y);
    vmax.z = max(vmax.z, v.z);
  }
  vmin -= Vector3(1, 1, 1);
  vmax += Vector3(1, 1, 1);
  
  container con(vmin.x, vmax.x, vmin.y, vmax.y, vmin.z, vmax.z, n_x, n_y, n_z,
                false, false, false, 8);

	for(int i = 0; i < (int)coord.size(); ++i) {
    const Vector3& v = coord[i];
		con.put(i, v.x, v.y, v.z);
	}

  voronoicell_neighbor c;

  neighbor_list.resize(n_part);
  neighbor_dist.resize(n_part);
  nei_area.resize(n_part);
  
  // Loop over all particles in the container and compute each Voronoi
  // cell
  c_loop_all cl(con);
  int dimension = 0;
  if(cl.start()) do if(con.compute_cell(c,cl)) {
        dimension+=1;
      } while (cl.inc());

  if(cl.start()) do if(con.compute_cell(c,cl)) {
        //cl.pos(x, y, z);
        int id = cl.pid();
        //cout << id << endl;
        if( id >= n_part )
          continue;

        vector<int> neigh;
        vector<double> areas;

        c.neighbors(neigh);
        c.face_areas(areas);

        neighbor_list[id] = neigh;
        nei_area[id] = areas;
        
      } while (cl.inc());

  for( int i = 0; i < n_part; ++i) {
    vector<int> nei_list = neighbor_list[i];
    vector<double> nei_dist;
    map<int, double> nei_area_temp;
    for( int jj = 0; jj < (int)nei_list.size(); ++jj) {
      int j = nei_list[jj];
      //# calc dist
      int i2 = i;
      int j2 = j % n_part;
      Vector3 delta = orig_coord[i2] - orig_coord[j2];
      if( abs(delta.x) > sim_box.x / 2 ) delta.x = abs(delta.x) - sim_box.x;
      if( abs(delta.y) > sim_box.y / 2 ) delta.y = abs(delta.y) - sim_box.y;
      if( abs(delta.z) > sim_box.z / 2 ) delta.z = abs(delta.z) - sim_box.z;
      double dist = delta.Norm();
      nei_dist.push_back(dist);

      nei_area_temp[j] = nei_area[i][jj];
    }

    nei_area[i].clear();
    misc::sort_by_distance(nei_list, nei_dist);
    for( int j : nei_list ) {
      nei_area[i].push_back(nei_area_temp[j]);
    }

    neighbor_list[i] = nei_list;
    neighbor_dist[i] = nei_dist;
  }

  for( int i = 0; i < n_part; ++i ) {
    for( int j = 0; j < (int)neighbor_list[i].size(); ++j ) {
      neighbor_list[i][j] = neighbor_list[i][j] % n_part;
    }
  }

  return;
}

void neighbor_build::build_neighbor_wrapper(vector<Vector3>& coord, const Vector3& box_length,
                                            const OP_SETTINGS& op_settings,
                                            map<string, vector<vector<int   >>>& neighbors_list,
                                            map<string, vector<vector<double>>>& neighbors_dist,
                                            map<string, vector<vector<double>>>& neighbors_area)
{
  neighbors_list.clear();
  neighbors_dist.clear();
  neighbors_area.clear();
  // calc initial thresh distance

  // neighbor or radius based neighbor list build
  if( !op_settings.neighbor.empty() || !op_settings.radius.empty() ) {
    double safe_factor = 1.7;
    double dist = 0.0;
    if( !op_settings.neighbor.empty() ) {
      auto it = max_element(op_settings.neighbor.begin(), op_settings.neighbor.end());
      dist = calc_thresh(box_length, coord.size(), *it, safe_factor);
    }
    
    if( !op_settings.radius.empty() ) {
      auto it = max_element(op_settings.radius.begin(), op_settings.radius.end());
      dist = max(*it, dist);
    }

    // calc initial neighbor list
    vector<vector<int   >> nei_list;
    vector<vector<double>> nei_dist;
    build_neighbor_list(coord, box_length, "thresh", dist, 0, nei_list, nei_dist);

    // cut up neighbor_list for small radius or num of neighbor.
    for( int i : op_settings.neighbor ) {
      bool done = false;
      while( done == false ) {
        done = true;
        try {
          vector<vector<int   >> new_list;
          vector<vector<double>> new_dist;
          neighbor_build::mod_neighbor_list(nei_list, nei_dist, i, 0, new_list, new_dist);
          string s = "N" + to_string(i);
          neighbors_list.insert(make_pair(s, new_list));
          neighbors_dist.insert(make_pair(s, new_dist));
        }
        catch(neighbor_build::SmallRadiusError& e) {
          // cutting up failed
          dist *= safe_factor;
          neighbor_build::build_neighbor_list(coord, box_length, "thresh", dist, 0, nei_list, nei_dist);
          done = false;
        }
      }
    }
    
    for( double i : op_settings.radius ) {
      char name[6];
      sprintf(name, "%03.2f", i);
      vector<vector<int   >> new_list;
      vector<vector<double>> new_dist;
      neighbor_build::mod_neighbor_list(nei_list, nei_dist, 0, i, new_list, new_dist);
      string s = string("R") + name;
      neighbors_list.insert(make_pair(s, new_list));
      neighbors_dist.insert(make_pair(s, new_dist));
    }
  }

  if( !op_settings.Delaunay.empty() ) {
    if( op_settings.neighbor.empty() && op_settings.radius.empty() ) {
      neighbors_list.clear();
      neighbors_dist.clear();
      neighbors_area.clear();
    }
    for( string i : op_settings.Delaunay ) {
      if( i == "standard" ) {
        vector<vector<int   >> nei_list;
        vector<vector<double>> nei_dist;
        vector<vector<double>> nei_area;
        build_neighbor_delaunay(coord, box_length, nei_list, nei_dist, nei_area);
        neighbors_list["Delaunay"] = nei_list;
        neighbors_dist["Delaunay"] = nei_dist;
        neighbors_area["Delaunay"] = nei_area;
      }
    }
  }

  return;
}
