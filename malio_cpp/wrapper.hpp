
#include <map>
#include "Vector3.hpp"
#include "Quaternion.hpp"
#include "ml.hpp"
#include "DataFrame.hpp"

using namespace std;

class ml_lsa
{
public:
  static map<string, vector<double>> op_analyze_with_neighbor_list(const vector<Vector3>& coord,
                                                                   const vector<Vector3>& direct,
                                                                   const Vector3& box_length,
                                                                   const string& NR_name,
                                                                   const OP_SETTINGS& op_settings,
                                                                   const vector<vector<int>>& n_list,
                                                                   const vector<vector<double>>& nei_area);
  static DataFrame op_analyze(const vector<Vector3>& _coord,
                              const vector<Quaternion>& direct,
                              const Vector3& box_length,
                              const OP_SETTINGS op_settings);
  static DataFrame op_analyze(const vector<Vector3>& coord,
                              const vector<Vector3>& direct,
                              const Vector3& box_length,
                              const OP_SETTINGS op_settings);
};

  
     
