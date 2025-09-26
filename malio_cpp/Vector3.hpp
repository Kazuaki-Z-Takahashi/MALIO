#pragma once

#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace std;

class Matrix33;

class Vector3 {
public:
  double x;
  double y;
  double z;

public:
  Vector3() : x(0.0), y(0.0), z(0.0) {}

  Vector3(double _x, double _y, double _z)
    : x(_x), y(_y), z(_z) {}

  Vector3(double d[3])
    : x(d[0]), y(d[1]), z(d[2]) {}

  void Assign(double _x, double _y, double _z) {
    x = _x;
    y = _y;
    z = _z;
  }

  double Norm2() const {
    return x * x + y * y + z * z;
  }
  
  double Norm() const {
    return sqrt(Norm2());
  }
  
  void Normalize() {
    double r = Norm2();
    if (r == 0.0) return;
    *this /= sqrt(r);
  }

  double& operator[](int i) {
    if(i == 0 ) return x;
    if(i == 1 ) return y;
    if(i == 2 ) return z;
    return x;
  }
  
  double operator[](int i) const {
    if(i == 0 ) return x;
    if(i == 1 ) return y;
    if(i == 2 ) return z;
    return 0.0;
  }
  
  Vector3 operator-() const {
    return Vector3(-x, -y, -z);
  }

  Vector3 operator+(const Vector3& v) const {
    Vector3 vr;
    vr.x = x + v.x;
    vr.y = y + v.y;
    vr.z = z + v.z;
    return vr;
  }

  Vector3 operator-(const Vector3& v) const {
    return *this + (-v);
  }

  Vector3 operator*(double d) const {
    Vector3 v;
    v.x = x * d;
    v.y = y * d;
    v.z = z * d;
    return v;
  }

  friend Vector3 operator*(double d, const Vector3& v)
  {
    return v * d;
  }

  Vector3 operator/(double d) const {
    d = 1.0 / d;
    return *this * d;
  }

  // Inner Product
  double operator*(const Vector3& v) const {
    return x * v.x + y * v.y + z * v.z;
  }

  // Outer Product
  Vector3 operator%(const Vector3& v) const {
    Vector3 vr;
    vr.x = y * v.z - z * v.y;
    vr.y = z * v.x - x * v.z;
    vr.z = x * v.y - y * v.x;
    return vr;
  }

  Vector3& operator=(const Vector3& v)
  {
    x = v.x;
    y = v.y;
    z = v.z;
    return *this;
  }
  
  Vector3& operator+=(const Vector3& v) {
    x = x + v.x;
    y = y + v.y;
    z = z + v.z;
    return *this;
  }

  Vector3& operator-=(const Vector3& v) {
    *this = *this - v;
    return *this;
  }

  Vector3& operator*=(double d) {
    *this = *this * d;
    return *this;
  }

  Vector3& operator/=(double d) {
    *this = *this / d;
    return *this;
  }

  bool operator==(const Vector3& v) const {
    if( x != v.x ) return false;
    if( y != v.y ) return false;
    if( z != v.z ) return false;
    return true;
  }

  bool operator!=(const Vector3& v) const {
    return !(*this == v);
  }
  
  void Print() const {
    cout << setprecision(16) << x << " " << y << " " << z << endl;
  }
};


class Matrix33 {
public:
  double a[3][3];

  Matrix33() {
    a[0][0] = 1.0; a[0][1] = 0.0; a[0][2] = 0.0;
    a[1][0] = 0.0; a[1][1] = 1.0; a[1][2] = 0.0;
    a[2][0] = 0.0; a[2][1] = 0.0; a[2][2] = 1.0;
  }

  Matrix33(
    double a00, double a01, double a02,
    double a10, double a11, double a12,
    double a20, double a21, double a22
    ) {
    a[0][0] = a00; a[0][1] = a01; a[0][2] = a02;
    a[1][0] = a10; a[1][1] = a11; a[1][2] = a12;
    a[2][0] = a20; a[2][1] = a21; a[2][2] = a22;
  }

  Matrix33(const Vector3& v0,
           const Vector3& v1,
           const Vector3& v2)
  {
    a[0][0] = v0.x; a[0][1] = v0.y; a[0][2] = v0.z;
    a[1][0] = v1.x; a[1][1] = v1.y; a[1][2] = v1.z;
    a[2][0] = v2.x; a[2][1] = v2.y; a[2][2] = v2.z;
  }

  Matrix33(const Matrix33& m) {
    a[0][0] = m.a[0][0]; a[0][1] = m.a[0][1]; a[0][2] = m.a[0][2];
    a[1][0] = m.a[1][0]; a[1][1] = m.a[1][1]; a[1][2] = m.a[1][2];
    a[2][0] = m.a[2][0]; a[2][1] = m.a[2][1]; a[2][2] = m.a[2][2];
  }
  
  double* operator[](int i) {
    return (double*)(a[i]);
  }
  
  Vector3 col(int i) const {
    return Vector3(a[0][i], a[1][i], a[2][i]);
  }
  
  Matrix33 operator-() const {
    Matrix33 m;
    m.a[0][0] = -a[0][0]; m.a[0][1] = -a[0][1]; m.a[0][2] = -a[0][2];
    m.a[1][0] = -a[1][0]; m.a[1][1] = -a[1][1]; m.a[1][2] = -a[1][2];
    m.a[2][0] = -a[2][0]; m.a[2][1] = -a[2][1]; m.a[2][2] = -a[2][2];
    return m;
  }
  
  Matrix33 operator+(const Matrix33& m) const {
    Matrix33 mr;
    mr.a[0][0] = a[0][0] + m.a[0][0];
    mr.a[0][1] = a[0][1] + m.a[0][1];
    mr.a[0][2] = a[0][2] + m.a[0][2];
    mr.a[1][0] = a[1][0] + m.a[1][0];
    mr.a[1][1] = a[1][1] + m.a[1][1];
    mr.a[1][2] = a[1][2] + m.a[1][2];
    mr.a[2][0] = a[2][0] + m.a[2][0];
    mr.a[2][1] = a[2][1] + m.a[2][1];
    mr.a[2][2] = a[2][2] + m.a[2][2];
    return mr;
  }
  
  Matrix33 operator-(const Matrix33& m) const {
    return *this + (-m);
  }

  Matrix33 operator*(const Matrix33& m) const {
    Matrix33 mr;
    mr.a[0][0] = a[0][0]*m.a[0][0] + a[0][1]*m.a[1][0] + a[0][2]*m.a[2][0];
    mr.a[0][1] = a[0][0]*m.a[0][1] + a[0][1]*m.a[1][1] + a[0][2]*m.a[2][1];
    mr.a[0][2] = a[0][0]*m.a[0][2] + a[0][1]*m.a[1][2] + a[0][2]*m.a[2][2];
    mr.a[1][0] = a[1][0]*m.a[0][0] + a[1][1]*m.a[1][0] + a[1][2]*m.a[2][0];
    mr.a[1][1] = a[1][0]*m.a[0][1] + a[1][1]*m.a[1][1] + a[1][2]*m.a[2][1];
    mr.a[1][2] = a[1][0]*m.a[0][2] + a[1][1]*m.a[1][2] + a[1][2]*m.a[2][2];
    mr.a[2][0] = a[2][0]*m.a[0][0] + a[2][1]*m.a[1][0] + a[2][2]*m.a[2][0];
    mr.a[2][1] = a[2][0]*m.a[0][1] + a[2][1]*m.a[1][1] + a[2][2]*m.a[2][1];
    mr.a[2][2] = a[2][0]*m.a[0][2] + a[2][1]*m.a[1][2] + a[2][2]*m.a[2][2];
    return mr;
  }

  Vector3 operator*(const Vector3& v) const {
    Vector3 vr;
    vr.x = a[0][0]*v.x + a[0][1]*v.y + a[0][2]*v.z;
    vr.y = a[1][0]*v.x + a[1][1]*v.y + a[1][2]*v.z;
    vr.z = a[2][0]*v.x + a[2][1]*v.y + a[2][2]*v.z;
    return vr;
  }

  Matrix33 operator*(double d) const {
    Matrix33 m;
    m.a[0][0] = a[0][0]*d; m.a[0][1] = a[0][1]*d; m.a[0][2] = a[0][2]*d; 
    m.a[1][0] = a[1][0]*d; m.a[1][1] = a[1][1]*d; m.a[1][2] = a[1][2]*d; 
    m.a[2][0] = a[2][0]*d; m.a[2][1] = a[2][1]*d; m.a[2][2] = a[2][2]*d; 
    return m;
  }

  Matrix33 operator/(double d) const {
    d = 1.0 / d;
    return *this * d;
  }

  void operator=(const Matrix33& m) {
    a[0][0] = m.a[0][0]; a[0][1] = m.a[0][1]; a[0][2] = m.a[0][2];
    a[1][0] = m.a[1][0]; a[1][1] = m.a[1][1]; a[1][2] = m.a[1][2];
    a[2][0] = m.a[2][0]; a[2][1] = m.a[2][1]; a[2][2] = m.a[2][2];
  }
  
  Matrix33& operator+=(const Matrix33& m) {
    *this = *this + m ;
    return *this;
  }

  Matrix33& operator-=(const Matrix33& m) {
    *this = *this - m ;
    return *this;
  }

  Matrix33& operator*=(double d) {
    *this = *this * d;
    return *this;
  }

  Matrix33& operator/=(double d) {
    *this = *this / d;
    return *this;
  }

  // v * m
  friend Vector3 operator*(const Vector3& v, const Matrix33& m)
  {
    Matrix33 mt = m;
    mt.Transpose();
    return mt * v;
  }

  
  void RotX(double angle) {
    double rad = M_PI * angle / 180.0;
    double s = sin(rad);
    double c = cos(rad);

    a[0][0] = 1; a[0][1] = 0; a[0][2] = 0;
    a[1][0] = 0; a[1][1] = c; a[1][2] = -s;
    a[2][0] = 0; a[2][1] = s; a[2][2] = c;
  }

  void RotY(double angle) {
    double radian = M_PI * angle / 180;
    double s = sin(radian);
    double c = cos(radian);

    a[0][0] = c;  a[0][1] = 0; a[0][2] = s;
    a[1][0] = 0;  a[1][1] = 1; a[1][2] = 0;
    a[2][0] = -s; a[2][1] = 0; a[2][2] = c;
  }

  void RotZ(double angle) {
    double radian = M_PI * angle / 180.0;
    double s = sin(radian);
    double c = cos(radian);

    a[0][0] = c;   a[0][1] = -s;  a[0][2] = 0.0;
    a[1][0] = s;   a[1][1] = c;   a[1][2] = 0.0;
    a[2][0] = 0.0; a[2][1] = 0.0; a[2][2] = 1.0;
  }

  void Scale(double sx, double sy, double sz) {
    a[0][0] = sx;  a[0][1] = 0.0; a[0][2] = 0.0;
    a[1][0] = 0.0; a[1][1] = sy;  a[1][2] = 0.0;
    a[2][0] = 0.0; a[2][1] = 0.0; a[2][2] = sz;
  }

  void Transpose() {
    double t00 = a[0][0], t01 = a[1][0], t02 = a[2][0];
    double t10 = a[0][1], t11 = a[1][1], t12 = a[2][1];
    double t20 = a[0][2], t21 = a[1][2], t22 = a[2][2];

    a[0][0] = t00; a[0][1] = t01; a[0][2] = t02;
    a[1][0] = t10; a[1][1] = t11; a[1][2] = t12;
    a[2][0] = t20; a[2][1] = t21; a[2][2] = t22;
  }

  void Zero() {
    a[0][0] = 0.0; a[0][1] = 0.0; a[0][2] = 0.0;
    a[1][0] = 0.0; a[1][1] = 0.0; a[1][2] = 0.0;
    a[2][0] = 0.0; a[2][1] = 0.0; a[2][2] = 0.0;
  }

  double Det() const {
    double det = 0.0;
    det += a[0][0]*a[1][1]*a[2][2] + a[1][0]*a[2][1]*a[0][2] + a[2][0]*a[0][1]*a[1][2];
    det -= a[0][0]*a[2][1]*a[1][2] + a[2][0]*a[1][1]*a[0][2] + a[1][0]*a[0][1]*a[2][2];
    return det;
  }

  bool Inverse() {
    double det = Det();
    if (det == 0.0) return false;

    double b00 = 0.0, b01 = 0.0, b02 = 0.0;
    double b10 = 0.0, b11 = 0.0, b12 = 0.0;
    double b20 = 0.0, b21 = 0.0, b22 = 0.0;

    b00 = a[1][1]*a[2][2] - a[1][2]*a[2][1];
    b01 = a[0][2]*a[2][1] - a[0][1]*a[2][2];
    b02 = a[0][1]*a[1][2] - a[0][2]*a[1][1];
    b10 = a[1][2]*a[2][0] - a[1][0]*a[2][2];
    b11 = a[0][0]*a[2][2] - a[0][2]*a[2][0];
    b12 = a[0][2]*a[1][0] - a[0][0]*a[1][2];
    b20 = a[1][0]*a[2][1] - a[1][1]*a[2][0];
    b21 = a[0][1]*a[2][0] - a[0][0]*a[2][1];
    b22 = a[0][0]*a[1][1] - a[0][1]*a[1][0];

    a[0][0] = b00; a[0][1] = b01; a[0][2] = b02;
    a[1][0] = b10; a[1][1] = b11; a[1][2] = b12;
    a[2][0] = b20; a[2][1] = b21; a[2][2] = b22;

    *this /= det;
    
    return true;
  }

  void Print() const {
    cout << a[0][0] << " " << a[0][1] << " " << a[0][2] << endl;
    cout << a[1][0] << " " << a[1][1] << " " << a[1][2] << endl;
    cout << a[2][0] << " " << a[2][1] << " " << a[2][2] << endl;
  }

};
