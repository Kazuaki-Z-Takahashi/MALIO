
#pragma once

#include <complex>
#include <vector>

using namespace std;

typedef complex<double> COMPLEX;

class SphericalHarmonics
{
public:
  static COMPLEX Y(int l, int m, double theta, double phi);
  static vector<COMPLEX> Yn(int l, double theta, double phi);

private:
  static double factorial(int n);
  static double P(int N, int M, double c, double s);
  static vector<double> Pn(int N, double c, double s);
  static double Theta(int l, int m, double theta);
  static vector<double> Thetan(int l, double theta);
  static COMPLEX Phi(int m, double phi);
};

