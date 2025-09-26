#pragma once

#include <cassert>
#include <iostream>

using namespace std;

struct int3
{
  int3(int _i1, int _i2, int _i3)
    : i1(_i1), i2(_i2), i3(_i3) {}
  int3(int _i[])
    : i1(_i[0]), i2(_i[1]), i3(_i[2]) {}
  int3()
    : i1(0), i2(0), i3(0) {}
  int3 operator-() const {
    return int3(-i1, -i2, -i3);
  }
  int3 operator+(const int3& i) const {
    return int3(i1 + i.i1, i2 + i.i2, i3 + i.i3);
  }
  int3 operator-(const int3& i) const {
    return -i + *this;
  }
  int3& operator+=(const int3& i) {
    i1 += i.i1;
    i2 += i.i2;
    i3 += i.i3;
    return *this;
  }
  int3 operator/(int n) const {
    return int3(i1 / n, i2 / n, i3 / n);
  }
  int operator*(const int3& i) const {
    return i1 * i.i1 + i2 * i.i2 + i3 * i.i3;
  }
  int& operator[](int i) {
    assert(0 <= 0 && i < 3);
    if( i == 0 ) return i1;
    if( i == 1 ) return i2;
    if( i == 2 ) return i3;
    return i1;
  }
  void Print() {
    cout << i1 << " " << i2 << " " << i3 << endl;
  }
  
  int i1;
  int i2;
  int i3;
};

