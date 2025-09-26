#pragma once

#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <iostream>
#include "Vector3.hpp"

using namespace std;

class Quaternion {
public:
  double w;
  Vector3 v;

public:
  Quaternion() : w(0.0) {}

  Quaternion(double _w, double _x, double _y, double _z)
    : w(_w)
  {
    v.x = _x;
    v.y = _y;
    v.z = _z;
  }

  Quaternion(double _w, const Vector3& _v)
    : w(_w), v(_v) {}
  
  Quaternion operator*(const Quaternion& q) const
  {
    Quaternion qr;
    qr.w = w * q.w - v * q.v;
    qr.v = w * q.v + q.w * v + v % q.v;
    return qr;
  }
  
  Vector3 Rotate(const Vector3& _v) const
  {
    Quaternion q = *this;
    Quaternion p(0.0, _v);
    Quaternion r(q.w, -q.v);
    p = q * p * r;
    return p.v;
  }
  
  void Print() const {
    cout << w << " " << v.x << " " << v.y << " " << v.z << endl;
  }
};

