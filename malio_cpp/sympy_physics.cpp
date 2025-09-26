
#include "sympy_physics.hpp"
#include <vector>
#include <cmath>
#include <functional>

using namespace std;

double sympy_physics::wigner_3j(int j_1, int j_2, int j_3, int m_1, int m_2, int m_3)
{
  if( m_1 + m_2 + m_3 != 0)
    return 0.0;
  int a1 = j_1 + j_2 - j_3;
  if( a1 < 0 )
    return 0.0;
  int a2 = j_1 - j_2 + j_3;
  if( a2 < 0 )
    return 0.0;
  int a3 = -j_1 + j_2 + j_3;
  if( a3 < 0 )
    return 0.0;
  if( abs(m_1) > j_1 || abs(m_2) > j_2 || abs(m_3) > j_3 )
    return 0.0;

  int maxfact = max({j_1 + j_2 + j_3 + 1, j_1 + abs(m_1), j_2 + abs(m_2), j_3 + abs(m_3)});

  vector<double> _Factlist(maxfact + 1);
  _Factlist[0] = 1.0;
  for(int i = 1; i <= maxfact; ++i) {
    _Factlist[i] = _Factlist[i-1] * i;
  }

  double argsqrt = 
    ( _Factlist[ j_1 + j_2 - j_3] *
      _Factlist[ j_1 - j_2 + j_3] *
      _Factlist[-j_1 + j_2 + j_3] *
      _Factlist[ j_1 - m_1      ] *
      _Factlist[ j_1 + m_1      ] *
      _Factlist[ j_2 - m_2      ] *
      _Factlist[ j_2 + m_2      ] *
      _Factlist[ j_3 - m_3      ] *
      _Factlist[ j_3 + m_3      ] ) /
    _Factlist[j_1 + j_2 + j_3 + 1];
  
  double ressqrt = sqrt(argsqrt);

  int imin = max({-j_3 + j_1 + m_2, -j_3 + j_2 - m_1, 0});
  int imax = min({j_2 + m_2, j_1 - m_1, j_1 + j_2 - j_3});

  double sumres = 0.0;
  for(int ii = imin; ii <= imax; ++ii) {
    double den =
      _Factlist[ii                   ] *
      _Factlist[ii  + j_3 - j_1 - m_2] *
      _Factlist[j_2 + m_2 - ii       ] *
      _Factlist[j_1 - ii  - m_1      ] *
      _Factlist[ii  + j_3 - j_2 + m_1] *
      _Factlist[j_1 + j_2 - j_3 - ii ];
    sumres += pow(-1, ii) / den;
  }
  
  int prefid = pow(-1, j_1 - j_2 - m_3);
  double res = ressqrt * sumres * prefid;

  return res;
}

