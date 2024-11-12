# -*- coding: utf-8 -*-

### import math
import numpy as np
import pyquaternion as pyquat
from malio_cy_def cimport *
cimport cython
from libcpp.pair cimport pair

cdef extern from "<algorithm>" namespace "std":
    void std_sort "std::sort" [iter](iter first, iter last)

cpdef double f_1_cy(double r):
    return r
@cython.cdivision(True)
cpdef double f_2_cy(double r):
    return 1/r

@cython.cdivision(True)
cpdef double f1(int j, dvec voronoi_area_list, dvec distance_list):
    cdef:
        double weight
        double dsum = 0.0
        int lvoronoi_area_list = len(voronoi_area_list)
        int i
    for i in range(lvoronoi_area_list):
        dsum += voronoi_area_list[i]
    ## weight = voronoi_area_list[j] / np.sum(voronoi_area_list)
    weight = voronoi_area_list[j] / dsum
    return weight
@cython.cdivision(True)
cpdef double f2(int j, dvec voronoi_area_list, dvec distance_list):
    cdef:
        double weight
        double ldistance_list = len(distance_list)
    ## weight = 1.0 / float(len(distance_list))
    weight = 1.0 / ldistance_list
    return weight
@cython.cdivision(True)
cpdef double f3(int j, dvec voronoi_area_list, dvec distance_list):
    cdef:
        double weight, sum_dist
        int ldistance_list = len(distance_list)
    sum_dist = 0
    for i in range(ldistance_list):
        sum_dist += 1.0/distance_list[i]
    ## weight = (1.0/distance_list[j]) / np.sum(sum_dist)
    weight = (1.0/distance_list[j]) / sum_dist
    return weight

cpdef double f_1(double r):
    return math.sqrt(10.*r)

cpdef double f_2(double r):
    return 10.*r

cpdef double f_3(double r):
    return 100.*r*r

@cython.cdivision(True)
cpdef double f_4(double r):
    return 1. - math.exp(-(10.*r - 3.0)**2 / (2.*0.15*0.15))

@cython.cdivision(True)
cpdef double f_5(double r):
    return 0.5 + 0.5 * math.exp(-(10.*r - 3.0)**2 / (2.*0.15*0.15))

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef double complex sph_harm_cy(int l_sph, int m_sph, double theta, double phi):
    cdef:
        double PI = math.acos(-1.0)
        double cosmphi = math.cos(phi*<double>m_sph)
        double sinmphi = math.sin(phi*<double>m_sph)
        double x = math.cos(theta)
        int l = l_sph
        int m = abs(m_sph)
        int i,ll
        double fact,oldfact,pll,pmm,pmmp1,omx2
        double complex I = 1j
    assert m >= 0 or m <= l or math.fabs(x) < 1.0, "bad argument for spherical harmonics"
    pmm=1.0
    if (m > 0):
        omx2=(1.0-x)*(1.0+x)
        fact=1.0
        for i in range(1,m+1):
            pmm *= omx2*fact/(fact+1.0)
            fact += 2.0

    pmm=math.sqrt((2*m+1)*pmm/(4.0*PI))
    if (m & 1):
        pmm=-pmm
    if (l == m):
        return pmm*(cosmphi+I*sinmphi)
    else:
        pmmp1=x*math.sqrt(2.0*m+3.0)*pmm
        if (l == (m+1)):
            return pmmp1*(cosmphi+I*sinmphi)
        else:
            oldfact=math.sqrt(2.0*m+3.0)
            for ll in range(m+2,l+1):
                fact=math.sqrt((4.0*ll*ll-1.0)/(ll*ll-m*m))
                pll=(x*pmmp1-pmm/oldfact)*fact
                oldfact=fact
                pmm=pmmp1
                pmmp1=pll
            return pll*(cosmphi+I*sinmphi)

@cython.boundscheck(False)
cpdef int ind_vdmax(vector[double] dvect) nogil:
    cdef:
        double vmax
        int i, maxi
    vmax = dvect[0]
    maxi = 0
    for i in range(dvect.size()):
        if dvect[i] > vmax:
            vmax = dvect[i]
            maxi = i
    return maxi

@cython.boundscheck(False)
cpdef double vdmax(vector[double] dvect) nogil:
    cdef:
        double vmax
        int i
    vmax = dvect[0]
    for i in range(dvect.size()):
        if dvect[i] > vmax:
            vmax = dvect[i]
    return vmax

@cython.cdivision(True)
cdef double generalbinom(double alpha,int k) :
    cdef:
        double tmp = 1.0, kk = <double>k
        int i
    for i in range(k):
        tmp = tmp * alpha/kk
        alpha -= 1.0
        kk -= 1.0
    return tmp

@cython.boundscheck(False)
cpdef legendre_coeff(int n_leg):
    cdef:
       double tmp, pfac = math.pow(2.0,n_leg),dn_leg, dk
       ### double *legcoeffs = <double*>malloc((n_leg+1)*sizeof(double))
       dvec legcoeffs = dvec(n_leg+1)
       int k

    dn_leg = <double>n_leg
    for k in range(n_leg+1):
        dk = <double>k
        tmp = generalbinom(dn_leg,k)*generalbinom((dn_leg+dk-1.0)*0.5,n_leg)
        legcoeffs[n_leg-k] = tmp*pfac

    return legcoeffs

def convert_3dim_to_1dim(array3d):
    array = np.zeros(3 * len(array3d))
    for i_i, _ in enumerate(array3d):
        for i_j in range(3):
            array[3 * i_i + i_j] = array3d[i_i][i_j]
    return array


""" quartation to direction 3d vector """


def q_to_xyz_org(q_list):
    xyz = []
    for i, _ in enumerate(q_list):
        now_q = q_list[i]
        quat = pyquat.Quaternion(
            x=now_q[1], y=now_q[2], z=now_q[3], w=now_q[0])
        xyz.append(quat.rotate([1, 0, 0]))
    return xyz


def q_to_xyz(q_list):
    xyz = []
    cdef:
        double q0, q1, q2, q3, ss, xx, yy, zz
        int i, l_q_list = len(q_list)
    for i in range(l_q_list):
        q0 = q_list[i][0]
        q1 = q_list[i][1]
        q2 = q_list[i][2]
        q3 = q_list[i][3]

        ss = q0*q0+q1*q1+q2*q2+q3*q3
        ss = 1.0/ss
        xx = 1.0-2.0*(q2*q2+q3*q3)*ss
        yy = 2.0*(q1*q2+q3*q0)*ss
        zz = 2.0*(q1*q3-q2*q0)*ss

        xyz.append([xx,yy,zz])
    return xyz


def vec_to_unit_vec(xyz):
    for i, _ in enumerate(xyz):
        length = np.linalg.norm(xyz[i])
        if length == 0:
            xyz[i][0] = 1.0
        else:
            xyz[i] = [xyz[i][j]/length for j in range(3)]
    return xyz


def add_index_to_list(nei_ij, dist_ij, size, dist, index):
    if len(nei_ij) < size:
        nei_ij.append(index)
        dist_ij.append(dist)
    else:
        if max(dist_ij) > dist:
            idx = dist_ij.index(max(dist_ij))
            nei_ij[idx] = index
            dist_ij[idx] = dist
    return [nei_ij, dist_ij]


def sort_by_distance(nei_list, nei_dist):
    for i, _ in enumerate(nei_list):
        temp_list = []
        temp_dist = []
        for j in range(len(nei_list[i])):
            idx = nei_dist[i].index(min(nei_dist[i]))
            temp_list.append(nei_list[i][idx])
            temp_dist.append(min(nei_dist[i]))
            nei_list[i].pop(idx)
            nei_dist[i].pop(idx)
        nei_list[i] = temp_list
        nei_dist[i] = temp_dist
    return [nei_list, nei_dist]

cpdef sort_by_distance_cy(vector[int] nei_list, vector[double] nei_dist):
    cdef:
        int i
        pair[double, int] entry
        vector[pair[double, int]] vpair
    for i in range(nei_list.size()):
        entry.first = nei_dist[i]
        entry.second = nei_list[i]
        vpair.push_back(entry)

    std_sort[vector[pair[double, int]].iterator](vpair.begin(), vpair.end())
    for i in range(nei_list.size()):
        nei_dist[i] = vpair[i].first
        nei_list[i] = vpair[i].second
    return nei_list, nei_dist

""" plane perpendicular to direction vector through point """


def gen_z_plane(point, direct):
    a_v = direct[0]
    b_v = direct[1]
    c_v = direct[2]
    d_v = -a_v * point[0] - b_v * point[1] - c_v * point[2]
    return [a_v, b_v, c_v, d_v]


def gen_neighbor_ij(coord_1d, direct_1d, args):
    [box_length, neighbor_list_ii, x_i, i_j, o_f, o_j, o_k, m_neighbor] = args

    x_j = calc_head_coordinate(coord_1d, direct_1d, i_j, o_f, o_j)

    i_j_nei = []
    i_j_dist = []
    for i_k in neighbor_list_ii:
        if i_j == i_k:
            continue
        x_k = calc_head_coordinate(
            coord_1d, direct_1d, i_k, o_f, o_k)
        dist = distance_ik_jk(x_i, x_j, box_length, x_k)
        [i_j_nei, i_j_dist] = add_index_to_list(
            i_j_nei, i_j_dist, m_neighbor, dist, i_k)

    [i_j_nei, i_j_dist] = sort_by_distance([i_j_nei], [i_j_dist])
    i_j_nei = i_j_nei[0]
    i_j_dist = i_j_dist[0]

    return i_j_nei


def gen_neighbor_ijk(coord_1d, direct_1d, args):
    [box_length, neighbor_list_ii, x_i, o_f, o_j, o_k, max_m] = args

    neighbor_ijk = []
    for i_j in neighbor_list_ii:
        args = [box_length, neighbor_list_ii,
                x_i, i_j, o_f, o_j, o_k, max_m]
        i_j_nei = gen_neighbor_ij(coord_1d, direct_1d, args)
        neighbor_ijk.append(i_j_nei)

    return neighbor_ijk


def v_neighb_ave(neighbor_list, val):
    if isinstance(val[0], type([])):
        val_ave = []  # [[1,2,3],[1,2,3], ... ]
        for i_i, _ in enumerate(neighbor_list):
            part = [0 for _ in range(len(val[i_i]))]
            for i_j in range(len(val[i_i])):
                for inei in neighbor_list[i_i] + [i_i]:
                    part[i_j] += val[inei][i_j] / float(
                        len(neighbor_list[i_i]) + 1)
            val_ave.append(part)
    elif isinstance(val[0], type(np.array([]))):
        val_ave = []  # [[1,2,3],[1,2,3], ... ]
        for i_i, _ in enumerate(neighbor_list):
            part = [0 for _ in range(len(val[i_i]))]
            for i_j in range(len(val[i_i])):
                for inei in neighbor_list[i_i] + [i_i]:
                    part[i_j] += val[inei][i_j] / float(
                        len(neighbor_list[i_i]) + 1)
            val_ave.append(part)
        val_ave = np.array(val_ave)
    else:
        val_ave = []
        for i_i, _ in enumerate(neighbor_list):
            part = []
            for inei in neighbor_list[i_i] + [i_i]:
                part.append(val[inei])
            val_ave.append(np.average(part))
    return val_ave


def move_vec(coord, direct, o_factor, orient, i_i):
    x_i = [coord[i_i][i] + o_factor * orient * direct[i_i][i]
           for i in range(3)]
    return x_i


def angle(v_1, v_2):
    return math.acos(np.dot(v_1, v_2) / (np.linalg.norm(v_1) * np.linalg.norm(v_2)))


def distance_ik_jk(x_i, x_j, box_length, x_k):
    x_ik = calc_delta(x_k, x_i, box_length)
    x_jk = calc_delta(x_k, x_j, box_length)
    distance = np.linalg.norm(x_ik) + np.linalg.norm(x_jk)
    return distance


def search_opposite_j_particle(coord_1d, direct_1d, neighbor_list_ii, x_i, i_j, x_j, box_length, o_f, o_k):
    x_i_j = calc_delta(x_j, x_i, box_length)
    x_j_opposite = [x_i[i] - x_i_j[i] for i in range(3)]

    nearest_i_k = 1000
    nearest_distnace = 10000000.0
    for i_k in neighbor_list_ii:
        if i_k == i_j:
            continue
        x_k = calc_head_coordinate(coord_1d, direct_1d, i_k, o_f, o_k)
        x_j_o_k = calc_delta(x_k, x_j_opposite, box_length)
        distance = np.linalg.norm(x_j_o_k)
        if distance <= nearest_distnace:
            nearest_distnace = distance
            nearest_i_k = i_k

    return nearest_i_k


def plane_point_distance(plane_var, box_length, point):
    a_v = plane_var[0]
    b_v = plane_var[1]
    c_v = plane_var[2]
    d_v = plane_var[3]

    distance = [0, 0, 0]
    for i_i in range(-1, 2):
        p_temp = [point[i] + i_i * box_length[i] for i in range(3)]
        x_v = p_temp[0]
        y_v = p_temp[1]
        z_v = p_temp[2]
        distance[i_i] = abs(a_v * x_v + b_v * y_v + c_v * z_v + d_v) / \
            math.sqrt(a_v**2 + b_v**2 + c_v**2)

    return min(distance)

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef dplane_point_distance(plane_var, box_length, point):
    cdef:
        double a_v = plane_var[0]
        double b_v = plane_var[1]
        double c_v = plane_var[2]
        double d_v = plane_var[3]
        double x_v, y_v, z_v, tmp
        double distance = 1.0e+12
        double dbox_length[3]
        double dpoint[3]
        int i_i, i

    for i in range(3):
        dbox_length[i] = box_length[i]
        dpoint[i] = point[i]

    for i_i in range(-1, 2):
        x_v = dpoint[0] + <double>i_i * dbox_length[0]
        y_v = dpoint[1] + <double>i_i * dbox_length[1]
        z_v = dpoint[2] + <double>i_i * dbox_length[2]
        tmp = math.fabs(a_v * x_v + b_v * y_v + c_v * z_v + d_v) / \
            math.sqrt(a_v**2 + b_v**2 + c_v**2)
        if tmp < distance:
            distance = tmp

    return distance

def calc_delta(x_end, x_start, box_length):
    cdef:
        int i
    delta = np.array([x_end[i] - x_start[i] for i in range(3)])
    for i in range(3):
        if delta[i] < -box_length[i] / 2.0:
            delta[i] += box_length[i]
        elif delta[i] >= box_length[i] / 2.0:
            delta[i] -= box_length[i]
    return delta


def convert_to_theta_phi(xyz):
    dist = np.linalg.norm(xyz)
    theta = math.acos(xyz[2] / dist)
    phi = math.atan2(xyz[1], xyz[0])

    return {'dist': dist, 'theta': theta, 'phi': phi}


def data_num_name_to_data_name_num(a_dict, num_part):
    # data[i_i][name] => data[name][i_i]
    b_dict = {}
    for name in sorted(a_dict[0]):
        a_temp = []
        for i_i in range(num_part):
            a_temp.append(a_dict[i_i][name])
        b_dict[name] = a_temp

    return b_dict


def calc_head_coordinate(coord_1d, direct_1d, i_i, o_f, o_i):
    coord_ii = [coord_1d[3 * i_i + i] for i in range(3)]
    direct_ii = [direct_1d[3 * i_i + i] for i in range(3)]
    x_i = move_vec([coord_ii], [direct_ii], o_f, o_i, 0)
    return x_i


def naming(mode, arg):
    if mode == 'a':
        [a_t, op_type, m_nei, o_f, o_i, o_j, o_k] = arg
        name = 'a=' + str(a_t) + '_type=' + str(op_type) + '_m=' + str(m_nei) + '_of=' + str(
            o_f) + '_oi=' + str(o_i) + '_oj=' + str(o_j) + '_ok=' + str(o_k)

    if mode == 'b':
        [a_t, m_fac, phi, n_pow, o_f, o_i, o_j, o_k] = arg
        name = 'a=' + str(a_t) + '_m=' + str(m_fac) + '_phi=' + str(
            phi) + '_n=' + str(n_pow) + '_of=' + str(o_f) + '_oi=' + str(
                o_i) + '_oj=' + str(o_j) + '_ok=' + str(o_k)

    if mode == 'c':
        [a_t, o_f, o_i, o_j, o_k] = arg
        name = 'a=' + str(a_t) + '_of=' + str(o_f) + '_oi=' + str(
            o_i) + '_oj=' + str(o_j) + '_ok=' + str(o_k)

    if mode == 'd':
        [a_t, o_f, o_i, o_j, o_k, f_1, f_2, f_3] = arg
        name = 'a=' + str(a_t) + '_of=' + \
            str(o_f) + '_oi=' + str(o_i) + \
            '_oj=' + str(o_j) + '_ok=' + str(o_k) + \
            '_f1=' + str(f_1) + '_f2=' + str(f_2) + '_f3=' + str(f_3)

    if mode == 'f':
        [a_t, o_f, o_i, o_j, o_k, f_1, f_2, l_nei] = arg
        name = 'a=' + str(a_t) + '_of=' + \
            str(o_f) + '_oi=' + str(o_i) + \
            '_oj=' + str(o_j) + '_ok=' + str(o_k) + \
            '_f1=' + str(f_1) + '_f2=' + str(f_2) + '_l=' + str(l_nei)

    if mode == 'h':
        [a_t, b_t, ibin, o_f, o_i, o_j, o_k] = arg
        name = 'a=' + str(a_t) + '_b=' + str(b_t) + '_bin=' + str(ibin) + '_of=' + str(
            o_f) + '_oi=' + str(o_i) + '_oj=' + str(o_j) + '_ok=' + str(o_k)

    if mode == 'q' or mode == 'w':
        [l_sph, a_t, b_t, o_f, o_i, o_j, p_weight] = arg
        name = 'l=' + str(l_sph) + '_a=' + str(a_t) + '_b=' + str(b_t) + '_of=' + str(
            o_f) + '_oi=' + str(o_i) + '_oj=' + str(o_j) + '_p=' + str(p_weight)

    if mode == 'q2' or mode == 'w2': 
        [l_sph, f_1, a_t, b_t, o_f, o_i, o_j] = arg
        name = 'l=' + str(l_sph) + '_f1=' + str(f_1) + \
            '_a=' + str(a_t) + '_b=' + str(b_t) +  \
            '_of=' + str(o_f) + '_oi=' + str(o_i) + '_oj=' + str(o_j)

    if mode == 's':
        [a_t, n_leg] = arg
        name = 'a=' + str(a_t) + '_n=' + str(n_leg)

    if mode == 't':
        [a_t, o_f, o_i, o_j, dist_layers, n_leg] = arg
        name = 'a=' + str(a_t) + '_n=' + str(n_leg) + '_of=' + str(
            o_f) + '_oi=' + str(o_i) + '_oj=' + str(
            o_j) + '_z=' + str(dist_layers)

    return name
