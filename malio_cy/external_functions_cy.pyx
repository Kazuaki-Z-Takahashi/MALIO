# -*- coding: utf-8 -*-
from malio_cy_def cimport *
cimport cython

@cython.boundscheck(False)
@cython.cdivision(True)
## cpdef double wigner_3j(int j_1, int j_2, int j_3, int m_1, int m_2, int m_3):
cpdef double f_1(double r):
    return math.sqrt(10.*r)

cpdef double f_2(double r):
    return 10.*r

cpdef f_3(double r):
    return 100.*r * r

cpdef f_4(double r):
    return 1.0 - math.exp(-(10.*r - 3.0)**2. / (2 * (0.15)**2))

cpdef f_5(double r):
    return 0.5 + 0.5 * math.exp(-(10.*r - 3.0)**2 / (2 * (0.15)**2))
