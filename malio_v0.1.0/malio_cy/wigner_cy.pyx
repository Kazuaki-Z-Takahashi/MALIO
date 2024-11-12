# -*- coding: utf-8 -*-
cimport cython
from malio_cy_def cimport *

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef double wigner_3j(int j_1, int j_2, int j_3, int m_1, int m_2, int m_3):
    """
    Calculate the Wigner 3j symbol `\operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)`.

    INPUT:

    -  ``j_1``, ``j_2``, ``j_3``, ``m_1``, ``m_2``, ``m_3`` - integer or half integer

    OUTPUT:

    Rational number times the square root of a rational number.

    Examples
    ========
    Calculate list of factorials::

        sage: from sage.functions.wigner import _calc_factlist
        sage: _calc_factlist(10)
        [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]

    >>> from sympy.physics.wigner import wigner_3j
    >>> wigner_3j(2, 6, 4, 0, 0, 0)
    sqrt(715)/143
    >>> wigner_3j(2, 6, 4, 0, 0, 1)
    0

    It is an error to have arguments that are not integer or half
    integer values::

        sage: wigner_3j(2.1, 6, 4, 0, 0, 0)
        Traceback (most recent call last):
        ...
        ValueError: j values must be integer or half integer
        sage: wigner_3j(2, 6, 4, 1, 0, -1.1)
        Traceback (most recent call last):
        ...
        ValueError: m values must be integer or half integer

    ALGORITHM:

    This function uses the algorithm of [Edmonds74]_ to calculate the
    value of the 3j symbol exactly. Note that the formula contains
    alternating sums over large factorials and is therefore unsuitable
    for finite precision arithmetic and only useful for a computer
    algebra system [Rasch03]_.

    AUTHORS:

    - Jens Rasch (2009-03-24): initial version
    """
    cdef:
        int ii, maxfact, a1, a2, a3, imax, imin
        double ressqrt, sumres, res, argsqrt, prefid, mm
        dvec _Factlist = dvec()

    _Factlist.push_back(1.0)
    ### _Factlist = [1.0]

    if (j_1 * 2) != j_1 * 2 or (j_2 * 2) != j_2 * 2 or \
            (j_3 * 2) != j_3 * 2:
        raise ValueError("j values must be integer or half integer")
    if (m_1 * 2) != m_1 * 2 or (m_2 * 2) != m_2 * 2 or \
            (m_3 * 2) != m_3 * 2:
        raise ValueError("m values must be integer or half integer")
    if m_1 + m_2 + m_3 != 0:
        return 0
    ### prefid = Integer((-1) ** int(j_1 - j_2 - m_3))
    ### prefid = math.pow(-1.0, j_1 - j_2 - m_3)
    if j_1 - j_2 - m_3 & 1:
        prefid = -1.0
    else:
        prefid = 1.0

    m_3 = -m_3
    a1 = j_1 + j_2 - j_3
    if a1 < 0:
        return 0
    a2 = j_1 - j_2 + j_3
    if a2 < 0:
        return 0
    a3 = -j_1 + j_2 + j_3
    if a3 < 0:
        return 0
    if (abs(m_1) > j_1) or (abs(m_2) > j_2) or (abs(m_3) > j_3):
        return 0

    maxfact = max(j_1 + j_2 + j_3 + 1, j_1 + abs(m_1), j_2 + abs(m_2),j_3 + abs(m_3))
    ### _calc_factlist(int(maxfact))
    if maxfact >= len(_Factlist):
        for ii in range(len(_Factlist), maxfact + 1):
            ### _Factlist.append(_Factlist[ii - 1] * <double>ii)
            _Factlist.push_back(_Factlist[ii - 1] * <double>ii)

    ### argsqrt = Integer(_Factlist[j_1 + j_2 - j_3] *
    argsqrt = (_Factlist[j_1 + j_2 - j_3] *
                     _Factlist[j_1 - j_2 + j_3] *
                     _Factlist[-j_1 + j_2 + j_3] *
                     _Factlist[j_1 - m_1] *
                     _Factlist[j_1 + m_1] *
                     _Factlist[j_2 - m_2] *
                     _Factlist[j_2 + m_2] *
                     _Factlist[j_3 - m_3] *
                     _Factlist[j_3 + m_3]) / \
        _Factlist[j_1 + j_2 + j_3 + 1]

    if argsqrt > 0:
        ressqrt = math.sqrt(<double>argsqrt)
    else:
        ressqrt = 0.0
    ### if ressqrt.is_complex or ressqrt.is_infinite:
    ###     ressqrt = ressqrt.as_real_imag()[0]

    imin = max(-j_3 + j_1 + m_2, -j_3 + j_2 - m_1, 0)
    imax = min(j_2 + m_2, j_1 - m_1, j_1 + j_2 - j_3)
    sumres = 0.0
    for ii in range(imin, imax + 1):
        if ii & 1:
           mm = -1.0
        else:
           mm = 1.0
        den = _Factlist[ii] * \
            _Factlist[ii + j_3 - j_1 - m_2] * \
            _Factlist[j_2 + m_2 - ii] * \
            _Factlist[j_1 - ii - m_1] * \
            _Factlist[ii + j_3 - j_2 + m_1] * \
            _Factlist[j_1 + j_2 - j_3 - ii]
        ### sumres = sumres + Integer((-1) ** ii) / den
        sumres = sumres + mm / den

    res = ressqrt * sumres * prefid
    return res
