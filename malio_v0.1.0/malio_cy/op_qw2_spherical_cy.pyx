# -*- coding: utf-8 -*-

from malio_cy import misc_cy as misc
from malio_cy.misc_cy import sph_harm_cy as sph_harm_cy
import numpy as np
## import math
from multiprocessing import Pool, Array
from scipy.special import sph_harm as _sph_harm
## from sympy.physics.wigner import wigner_3j
from malio_cy.wigner_cy import *
cimport cython
from malio_cy_def cimport *

def func_to_value_wigner(l_sph, wigner3j, func):
    sum_vec = 0.0
    for m1 in range(-l_sph, l_sph+1):
        for m2 in range(-l_sph, l_sph+1):
            m3 = -m1 - m2
            if -l_sph <= m3 and m3 <= l_sph:
                wig = wigner3j[m1][m2][m3]
                sum_vec += wig * np.real(func[m1]*func[m2]*func[m3])

    sum_norm2 = 0.0
    for i_j in range(-l_sph, l_sph + 1):
        comp = func[i_j]
        sum_norm2 += np.real(comp*np.conjugate(comp))
    sum_norm = pow(sum_norm2, 3.0/2.0)

    w_value = np.real(sum_vec) / sum_norm
    return round(w_value, 14)

def gen_wigner3j(l_sph):
    l2 = 2*l_sph + 1
    wig = [[[0.0 for m1 in range(l2)] for m2 in range(l2)] for m3 in range(l2)]
    for m1 in range(-l_sph, l_sph+1):
        for m2 in range(-l_sph, l_sph+1):
            m3 = -m1 - m2
            if -l_sph <= m3 and m3 <= l_sph:
                wig[m1][m2][m3] = float(
                    wigner_3j(l_sph, l_sph, l_sph, m1, m2, m3))
    return wig

def sph_harm(l_sph, m_sph, theta, phi):
    return _sph_harm(m_sph, l_sph, phi, theta)

def calc_q_weighted(l_sph, theta, phi, weight):
    q_l = [0 for i in range(2 * l_sph + 1)]
    for m_sph in range(l_sph + 1):
        q_l[m_sph] = weight*sph_harm(l_sph, m_sph, theta, phi)
    for m_sph in range(-l_sph, 0):
        q_l[m_sph] = weight*((-1)**m_sph) * \
            np.conjugate(sph_harm(l_sph, m_sph, theta, phi))
    return q_l

def func_to_value(l_sph, func):
    sum_norm2 = 0.0
    for i_j in range(2 * l_sph + 1):
        comp = func[i_j]
        sum_norm2 += np.real(comp * np.conjugate(comp))
    q_value = math.sqrt(sum_norm2 * (4.0 * math.pi) / (2.0 * l_sph + 1.0))
    return round(q_value, 14)

def calc_q_body(coord_ii, direct_ii, coord_ij, direct_ij, args):
    [box_length, neighbor_list_ii, i_i, l_list, nei_area,
     func_list, o_factor, oi_oj] = args

    comb = [(l_sph, func, o_f, o_i, o_j)
            for l_sph in l_list for func in func_list
            for o_f in o_factor for o_i in oi_oj for o_j in oi_oj]

    q_func_temp = {}
    for l_sph, func, o_f, o_i, o_j in comb:
        name = misc.naming('q2', [l_sph, func.__name__, 0, 0, o_f, o_i, o_j])

        x_i = misc.move_vec([coord_ii], [direct_ii], o_f, o_i, 0)

        dist_list = []
        for j in range(len(neighbor_list_ii)):
            x_j = misc.move_vec([coord_ij[j]], [direct_ij[j]], o_f, o_i, 0)
            delta = misc.calc_delta(x_i, x_j, box_length)
            dist_list.append(np.linalg.norm(delta))

        # neighbor
        q_temp = np.array([0 + 0j for i in range(2 * l_sph + 1)])
        for j in range(len(neighbor_list_ii)):
            x_j = misc.move_vec([coord_ij[j]], [direct_ij[j]], o_f, o_i, 0)
            delta = misc.calc_delta(x_i, x_j, box_length)
            pol = misc.convert_to_theta_phi(delta)
            weight = func(j, nei_area, dist_list)
            q_l = calc_q_weighted(l_sph, pol['theta'], pol['phi'], weight)

            for i_k in range(2 * l_sph + 1):
                q_temp[i_k] += q_l[i_k]

        q_func_temp[name] = q_temp

    return q_func_temp

def calc_q_wrapper(args):
    [box_length, neighbor_list_ii, i_i, l_list, nei_area,
     func_list, o_factor, oi_oj] = args

    coord_ii = [coord_1d[3 * i_i + i] for i in range(3)]
    direct_ii = [direct_1d[3 * i_i + i] for i in range(3)]

    coord_ij = []
    direct_ij = []
    for j in range(len(neighbor_list_ii)):
        i_j = neighbor_list_ii[j]
        coord_ij.append([coord_1d[3*i_j + i] for i in range(3)])
        direct_ij.append([direct_1d[3*i_j + i] for i in range(3)])

    q_func_temp = calc_q_body(coord_ii, direct_ii, coord_ij, direct_ij, args)
    return q_func_temp

@cython.boundscheck(False)
@cython.cdivision(True)
def calc_spherical_order_parameter(calc_type, coord, direct, box_length, setting, neighbor_list, neighbor_area, thread_num):
    # [Q_N]_l_a_b_oi_oj_P
    a_times = setting['ave_times']
    b_times = setting['b_in_Q']
    oi_oj = setting['oi_oj']
    o_factor = setting['o_factor']
    l_list = setting['l_in_Q']
    func_list = setting['function_in_Q2']

    # calc spherical function
    # prepare parallel
    cdef:
        int i_i, i_j, i_k, len_c = len(coord), o_f, o_i, o_j, i, j
        int l_neighbor_list_ii, i_2, l_sph, m_sph, p_weight, l_sph2p1
        int l_q_func_name_old_0, m1, m2, m3, mm1, mm2, mm3, li_j, lwigner3j
        double dist, theta, phi, inv_l_sph2p1, sum_norm2, sum_vec
        double q_value, PI4 = 4.0*math.acos(-1.0), dl_neighbor_list_ii_1, dpart
        double x_i[3]
        double x_j[3]
        double delta[3]
        vector[double] dbox_length = box_length
        vector[vector[double]] dcoord = coord
        vector[vector[double]] ddirect = direct
        vector[vector[double]] dneighbor_area = neighbor_area
        vector[vector[int]] ineighbor_list = neighbor_list
        ivec im1 = ivec()
        ivec im2 = ivec()
        ivec im3 = ivec()
        ## dvec w_val = dvec()
        dvec dwigner3j = dvec()
        dvec ddist_list = dvec()
        dvec q_val = dvec()
        cvec q_temp = cvec()
        cvec cpart = cvec()
        cvec_vec q_func_name = cvec_vec()
        cvec_vec q_func_name_old = cvec_vec()
        double complex sphharm, part_temp, cl_neighbor_list_ii_1, I = 1j, comp
        double weight

    ## global coord_1d
    ## coord_1d = Array('d', misc.convert_3dim_to_1dim(coord), lock=False)

    ## global direct_1d
    ## direct_1d = Array('d', misc.convert_3dim_to_1dim(direct), lock=False)

    ## now_pool = Pool(thread_num)
    args = [[box_length, neighbor_list[i_i], i_i, l_list, neighbor_area[i_i],
             func_list, o_factor, oi_oj] for i_i in range(len(coord))]
    ## q_func_temp = now_pool.map(calc_q_wrapper, args)
    q_func_temp= []
    ## for arg in args:
    ##     res = calc_q_wrapper(arg)
    ## [box_length, neighbor_list[i_i], i_i, l_list, neighbor_area[i_i], func_list, o_factor, oi_oj]
    ## [box_length, neighbor_list_ii, i_i, l_list, nei_area, func_list, o_factor, oi_oj]

    for i_i in range(len_c):
        l_neighbor_list_ii = len(neighbor_list[i_i])

        #coord_ii = [coord_1d[3 * i_i + i] for i in range(3)]
        #direct_ii = [direct_1d[3 * i_i + i] for i in range(3)]
        #coord_ij = []
        #direct_ij = []
        #for j in range(l_neighbor_list_ii):
        #    i_j = neighbor_list[i_i][j]
        #    coord_ij.append([coord_1d[3*i_j + i] for i in range(3)])
        #    direct_ij.append([direct_1d[3*i_j + i] for i in range(3)])

        comb = [(l_sph, func, o_f, o_i, o_j)
                for l_sph in l_list for func in func_list
                for o_f in o_factor for o_i in oi_oj for o_j in oi_oj]

        res = {}
        for l_sph, func, o_f, o_i, o_j in comb:
            name = misc.naming('q2', [l_sph, func.__name__, 0, 0, o_f, o_i, o_j])
            l_sph2p1 = 2*l_sph+1
            ## x_i = misc.move_vec([coord_ii], [direct_ii], o_f, o_i, 0)
            for i in range(3):
                x_i[i] = dcoord[i_i][i] +  <double>o_f *  <double>o_i * ddirect[i_i][i]
            ## dist_list = []
            ddist_list = dvec(l_neighbor_list_ii)
            for j in range(l_neighbor_list_ii):
                i_j = ineighbor_list[i_i][j]
                ## x_j = misc.move_vec([coord_ij[j]], [direct_ij[j]], o_f, o_i, 0)
                for i in range(3):
                    x_j[i] = dcoord[i_j][i] +  <double>o_f *  <double>o_i * ddirect[i_j][i]
                ## delta = misc.calc_delta(x_i, x_j, box_length)
                for i in range(3):
                    delta[i] = x_i[i] - x_j[i]
                    if delta[i] < -dbox_length[i] * 0.5:
                        delta[i] += dbox_length[i]
                    elif delta[i] >= dbox_length[i] * 0.5:
                        delta[i] -= dbox_length[i]
                dist = 0.0
                for i in range(3):
                    dist += delta[i]*delta[i]
                dist = math.sqrt(dist)
                ## dist_list.append(np.linalg.norm(delta))
                ddist_list[j] = dist
            # neighbor
            ## q_temp = np.array([0 + 0j for i in range(2 * l_sph + 1)])
            q_temp = cvec(l_sph2p1)
            ## for j in range(len(neighbor_list[i_i])):
            for j in range(l_neighbor_list_ii):
                i_j = ineighbor_list[i_i][j]
                ## x_j = misc.move_vec([coord_ij[j]], [direct_ij[j]], o_f, o_i, 0)
                for i in range(3):
                    x_j[i] = dcoord[i_j][i] +  <double>o_f *  <double>o_i * ddirect[i_j][i]
                ## delta = misc.calc_delta(x_i, x_j, box_length)
                for i in range(3):
                    delta[i] = x_i[i] - x_j[i]
                    if delta[i] < -dbox_length[i] * 0.5:
                        delta[i] += dbox_length[i]
                    elif delta[i] >= dbox_length[i] * 0.5:
                        delta[i] -= dbox_length[i]
                ## pol = misc.convert_to_theta_phi(delta)
                dist = 0.0
                for i in range(3):
                    dist += delta[i]*delta[i]
                dist = math.sqrt(dist)
                theta = math.acos(delta[2] / dist)
                phi = math.atan2(delta[1], delta[0])
                ## weight = func(j, dneighbor_area[i_i], dist_list)
                weight = func(j, dneighbor_area[i_i], ddist_list)
                ## q_l = calc_q_weighted(l_sph, pol['theta'], pol['phi'], weight)
                ## for i_k in range(2 * l_sph + 1):
                ##     q_temp[i_k] += q_l[i_k]
                q_temp[0] += weight*sph_harm_cy(l_sph, 0, theta, phi)
                for m_sph in range(1, l_sph + 1):
                    sphharm = sph_harm_cy(l_sph, m_sph, theta, phi)
                    q_temp[m_sph] += weight*sphharm
                    q_temp[l_sph2p1-m_sph] += weight*sphharm
            res[name] = q_temp

        q_func_temp.append(res)
    ## now_pool.close()

    ## del coord_1d
    ## del direct_1d

    q_func = misc.data_num_name_to_data_name_num(q_func_temp, len(coord))

    # calc function average
    comb = [(l_sph, func, b_t, o_f, o_i, o_j)
            for l_sph in l_list
            for func in func_list
            for b_t in range(b_times)
            for o_f in o_factor
            for o_i in oi_oj for o_j in oi_oj]

    for l_sph, func, b_t, o_f, o_i, o_j in comb:
        name = misc.naming('q2', [l_sph, func.__name__, 0, b_t+1, o_f, o_i, o_j])
        name_old = misc.naming('q2', [l_sph, func.__name__, 0, b_t, o_f, o_i, o_j])
        l_q_func_name_old_0 = len(q_func[name_old][0])
        q_func_name_old = cvec_vec(len_c,cvec(l_q_func_name_old_0))
        for i in range(len_c):
            for j in range(l_q_func_name_old_0):
                q_func_name_old[i][j] = q_func[name_old][i][j]
        ## q_func[name] = misc.v_neighb_ave(neighbor_list, q_func[name_old])
        q_func[name] = []  # [[1,2,3],[1,2,3], ... ]
        for i_i in range(len_c):
            l_neighbor_list_ii = len(neighbor_list[i_i])
            cl_neighbor_list_ii_1 = <double complex>(l_neighbor_list_ii + 1)
            ### part = [0 for i in range(l_q_func_name_old_0)]
            cpart = cvec(l_q_func_name_old_0)
            for i_j in range(l_q_func_name_old_0):
                part_temp = 0.0 + I*0.0
                for ii_j in range(l_neighbor_list_ii):
                    inei = ineighbor_list[i_i][ii_j]
                    part_temp += q_func_name_old[inei][i_j]
                part_temp += q_func_name_old[i_i][i_j]
                ### part.append(part_temp/cl_neighbor_list_ii_1)
                cpart[i_j] = part_temp/cl_neighbor_list_ii_1
            ### q_func[name].append(part)
            q_func[name].append(cpart)

    # func to value
    comb = [(func, o_f, o_i, o_j)
            for func in func_list
            for o_f in o_factor
            for o_i in oi_oj for o_j in oi_oj]
    op_data = {}
    if calc_type == 'Q':
        # comb = [(l_sph, func, o_f, o_i, o_j)
        #         for l_sph in l_list
        #         for func in func_list
        #         for o_f in o_factor
        #         for o_i in oi_oj for o_j in oi_oj]
        # op_data = {}
        for l_sph in l_list:
            l_sph2p1 = 2*l_sph+1
            inv_l_sph2p1 = 1.0/<double>l_sph2p1
            for b_t in range(b_times + 1):
                for func, o_f, o_i, o_j in comb:
                    name = misc.naming('q2', [l_sph, func.__name__, 0, b_t, o_f, o_i, o_j])
                    ## q_val = []
                    ## for i_i in range(len(coord)):
                    ##     q_val.append(func_to_value(l_sph, q_func[name][i_i]))
                    q_val = dvec()
                    for i_i in range(len_c):
                        sum_norm2 = 0.0
                        for i_j in range(l_sph2p1):
                            comp = q_func[name][i_i][i_j]
                            ## sum_norm2 += np.real(comp * np.conjugate(comp))
                            sum_norm2 += (comp*comp.conjugate()).real
                        q_value = math.sqrt(sum_norm2*PI4*inv_l_sph2p1)
                        q_value = round(q_value, 14)
                        q_val.push_back(q_value)
                    op_data[name] = q_val
    elif calc_type == 'W':
        ## comb = [(func, o_f, o_i, o_j)
        ##         for func in func_list
        ##         for o_f in o_factor
        ##         for o_i in oi_oj for o_j in oi_oj]
        ## op_data = {}
        for l_sph in l_list:
            ## wigner3j = gen_wigner3j(l_sph)
            l_sph2p1 = 2*l_sph+1
            dwigner3j = dvec()
            im1 = ivec()
            im2 = ivec()
            im3 = ivec()
            lwigner3j = 0
            for m1 in range(-l_sph, l_sph+1):
                for m2 in range(-l_sph, l_sph+1):
                    m3 = -m1 - m2
                    if -l_sph <= m3 and m3 <= l_sph:
                        im1.push_back((m1+l_sph2p1)%l_sph2p1)
                        im2.push_back((m2+l_sph2p1)%l_sph2p1)
                        im3.push_back((m3+l_sph2p1)%l_sph2p1)
                        dwigner3j.push_back(wigner_3j(l_sph,l_sph,l_sph,m1,m2,m3))
                        lwigner3j += 1
            for b_t in range(b_times + 1):
                for func, o_f, o_i, o_j in comb:
                    name = misc.naming('w2', [l_sph, func.__name__, 0, b_t, o_f, o_i, o_j])
                    w_val = []
                    q_func_name = cvec_vec(len_c,cvec(l_sph2p1))
                    for i in range(len_c):
                        for j in range(l_sph2p1):
                            q_func_name[i][j] = q_func[name][i][j]
                    for i_i in range(len_c):
                        ## w_val.append(func_to_value_wigner(l_sph, wigner3j, q_func[name][i_i]))
                        sum_vec = 0.0
                        for i in range(lwigner3j):
                            sum_vec += dwigner3j[i]* \
                            (q_func_name[i_i][im1[i]]*q_func_name[i_i][im2[i]]*q_func_name[i_i][im3[i]]).real
                        sum_norm2 = 0.0
                        for i_j in range(-l_sph, l_sph + 1):
                            li_j = (i_j+l_sph2p1) % l_sph2p1
                            comp = q_func_name[i_i][li_j]
                            sum_norm2 += (comp*comp.conjugate()).real
                        sum_norm = math.pow(sum_norm2, 1.5)
                        w_val.append(round(sum_vec/sum_norm, 14))
                    op_data[name] = w_val

    # neighbor value averaging
    comb = [(l_sph, func, a_t, b_t, o_f, o_i, o_j)
            for l_sph in l_list
            for func in func_list
            for a_t in range(a_times)
            for b_t in range(b_times+1)
            for o_f in o_factor for o_i in oi_oj for o_j in oi_oj]
    for l_sph, func, a_t, b_t, o_f, o_i, o_j in comb:
        name = misc.naming(
            'q2', [l_sph, func.__name__, a_t+1, b_t, o_f, o_i, o_j])
        name_old = misc.naming(
            'q2', [l_sph, func.__name__, a_t, b_t, o_f, o_i, o_j])
        ## op_data[name] = misc.v_neighb_ave(neighbor_list, op_data[name_old])
        ## val_ave = []
        q_val = dvec()
        for i_i in range(len_c):
            l_neighbor_list_ii = len(neighbor_list[i_i])
            dl_neighbor_list_ii_1 = <double>(l_neighbor_list_ii + 1)
            dpart = 0.0
            for ii_j in range(l_neighbor_list_ii):
                inei = ineighbor_list[i_i][ii_j]
                dpart += op_data[name_old][inei]
                ## part.append(op_data[name_old][inei])
            dpart += op_data[name_old][i_i]
            ## val_ave.append(np.average(part))
            q_val.push_back(dpart/dl_neighbor_list_ii_1)
        ## op_data[name] = val_ave
        op_data[name] = q_val

    return op_data

def calc_spherical_order_parameter_org(calc_type, coord, direct, box_length, setting, neighbor_list, neighbor_area, thread_num):
    # [Q_N]_l_a_b_oi_oj_P
    a_times = setting['ave_times']
    b_times = setting['b_in_Q']
    oi_oj = setting['oi_oj']
    o_factor = setting['o_factor']
    l_list = setting['l_in_Q']
    func_list = setting['function_in_Q2']

    # calc spherical function
    # prepare parallel
    global coord_1d
    coord_1d = Array('d', misc.convert_3dim_to_1dim(coord), lock=False)

    global direct_1d
    direct_1d = Array('d', misc.convert_3dim_to_1dim(direct), lock=False)

    now_pool = Pool(thread_num)
    args = [[box_length, neighbor_list[i_i], i_i, l_list, neighbor_area[i_i],
             func_list, o_factor, oi_oj] for i_i in range(len(coord))]
    q_func_temp = now_pool.map(calc_q_wrapper, args)
    now_pool.close()

    del coord_1d
    del direct_1d

    q_func = misc.data_num_name_to_data_name_num(q_func_temp, len(coord))

    # calc function average
    comb = [(l_sph, func, b_t, o_f, o_i, o_j)
            for l_sph in l_list
            for func in func_list
            for b_t in range(b_times)
            for o_f in o_factor
            for o_i in oi_oj for o_j in oi_oj]

    for l_sph, func, b_t, o_f, o_i, o_j in comb:
        name = misc.naming(
            'q2', [l_sph, func.__name__, 0, b_t+1, o_f, o_i, o_j])
        name_old = misc.naming(
            'q2', [l_sph, func.__name__, 0, b_t, o_f, o_i, o_j])
        q_func[name] = misc.v_neighb_ave(neighbor_list, q_func[name_old])

    # func to value
    if calc_type == 'Q':
        comb = [(l_sph, func, o_f, o_i, o_j)
                for l_sph in l_list
                for func in func_list
                for o_f in o_factor
                for o_i in oi_oj for o_j in oi_oj]
        op_data = {}
        for b_t in range(b_times + 1):
            for l_sph, func, o_f, o_i, o_j in comb:
                name = misc.naming(
                    'q2', [l_sph, func.__name__, 0, b_t, o_f, o_i, o_j])
                q_val = []
                for i_i in range(len(coord)):
                    q_val.append(func_to_value(l_sph, q_func[name][i_i]))
                op_data[name] = q_val
    elif calc_type == 'W':
        comb = [(func, o_f, o_i, o_j)
                for func in func_list
                for o_f in o_factor
                for o_i in oi_oj for o_j in oi_oj]

        op_data = {}
        for l_sph in l_list:
            wigner3j = gen_wigner3j(l_sph)
            for b_t in range(b_times + 1):
                for func, o_f, o_i, o_j in comb:
                    name = misc.naming(
                        'w2', [l_sph, func.__name__, 0, b_t, o_f, o_i, o_j])
                    w_val = []
                    for i_i in range(len(coord)):
                        w_val.append(func_to_value_wigner(
                            l_sph, wigner3j, q_func[name][i_i]))
                    op_data[name] = w_val

    # neighbor value averaging
    comb = [(l_sph, func, a_t, b_t, o_f, o_i, o_j)
            for l_sph in l_list
            for func in func_list
            for a_t in range(a_times)
            for b_t in range(b_times+1)
            for o_f in o_factor for o_i in oi_oj for o_j in oi_oj]
    for l_sph, func, a_t, b_t, o_f, o_i, o_j in comb:
        name = misc.naming(
            'q2', [l_sph, func.__name__, a_t+1, b_t, o_f, o_i, o_j])
        name_old = misc.naming(
            'q2', [l_sph, func.__name__, a_t, b_t, o_f, o_i, o_j])
        op_data[name] = misc.v_neighb_ave(neighbor_list, op_data[name_old])

    return op_data

def spherical_order_parameter(coord, direct, box_length, setting, neighbor_list, neighbor_area, thread_num):
    op_data = calc_spherical_order_parameter(
        'Q', coord, direct, box_length, setting, neighbor_list, neighbor_area, thread_num)
    return op_data

def w_order_parameter(coord, direct, box_length, setting, neighbor_list, neighbor_area, thread_num):
    op_data = calc_spherical_order_parameter(
        'W', coord, direct, box_length, setting, neighbor_list, neighbor_area, thread_num)
    return op_data
