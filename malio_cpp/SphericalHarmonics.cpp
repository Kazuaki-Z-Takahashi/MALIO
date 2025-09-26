#include "SphericalHarmonics.hpp"

const static COMPLEX I(0.0, 1.0);

double SphericalHarmonics::factorial(int n)
{
  double d = 1.0;
  for(int i = 2; i <= n; ++i) {
    d *= i;
  }
  return d;
}

double SphericalHarmonics::P(int N, int M, double c, double s)
{
  if( N == 0 )
    return 1.0;
  
  vector<vector<double>> p(N+1, vector<double>(N+1, 0.0));

  p[0][0] = 1.0;
  p[1][0] = c;
  p[1][1] = s;

  for(int n = 2; n <= N; ++n) {
    p[n][n] = p[n-1][n-1] * s * (2 * n - 1);
    for(int m = n - 1; m > 0; --m) {
      p[n][m] = c * p[n-1][m] + (n + m - 1) * s * p[n-1][m-1];
    }
    p[n][0] = ((2 * n - 1) * c * p[n-1][0] - (n - 1) * p[n-2][0]) / n;
  }
  
  return p[N][M];
}

vector<double> SphericalHarmonics::Pn(int N, double c, double s)
{
  if( N == 0 )
    return vector<double>(1, 1.0);
  
  vector<vector<double>> p(N+1, vector<double>(N+1, 0.0));

  p[0][0] = 1.0;
  p[1][0] = c;
  p[1][1] = s;

  for(int n = 2; n <= N; ++n) {
    p[n][n] = p[n-1][n-1] * s * (2 * n - 1);
    for(int m = n - 1; m > 0; --m) {
      p[n][m] = c * p[n-1][m] + (n + m - 1) * s * p[n-1][m-1];
    }
    p[n][0] = ((2 * n - 1) * c * p[n-1][0] - (n - 1) * p[n-2][0]) / n;
  }

  return p[N];
}

double SphericalHarmonics::Theta(int l, int m, double theta)
{
  int ma = abs(m);
  double d1 = (2 * l + 1) * factorial(l - ma);
  double d2 =  2          * factorial(l + ma);
  double d = pow(-1, (m + ma) / 2) * sqrt(d1 / d2) * P(l, ma, cos(theta), abs(sin(theta)));
  return d;
}

vector<double> SphericalHarmonics::Thetan(int l, double theta)
{
  vector<double> vd = Pn(l, cos(theta), abs(sin(theta)));

  for(int m = 0; m <= l; ++m) {
    double d1 = (2 * l + 1) * factorial(l - m);
    double d2 =  2          * factorial(l + m);
    vd[m] *= pow(-1, m) * sqrt(d1 / d2);
  }
  
  return vd;
}

COMPLEX SphericalHarmonics::Phi(int m, double phi)
{
  const double d = 1.0 / sqrt(2.0 * M_PI);
  return d * exp(I * (m * phi));
}

COMPLEX SphericalHarmonics::Y(int l, int m, double theta, double phi)
{
  return Theta(l, m, theta) * Phi(m, phi);
}

vector<COMPLEX> SphericalHarmonics::Yn(int l, double theta, double phi)
{
  vector<double> vd = Thetan(l, theta);
  vector<COMPLEX> vc(l + 1);
  for(int m = 0; m <= l; ++m) {
    vc[m] = vd[m] * Phi(m, phi);
  }
  return vc;
}

