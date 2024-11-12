# -*- coding: utf-8 -*-

import misc_cy
import numpy as np
### import math
from multiprocessing import Pool, Array
from scipy.special import sph_harm as _sph_harm
cimport cython
from malio_cy_def cimport *


def sph_harm(l_sph, m_sph, theta, phi):
    return _sph_harm(m_sph, l_sph, phi, theta)


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


def calc_q(l_sph, theta, phi):
    q_l = [0 for i in range(2 * l_sph + 1)]
    for m_sph in range(l_sph + 1):
        q_l[m_sph] = sph_harm(l_sph, m_sph, theta, phi)
    for m_sph in range(-l_sph, 0):
        q_l[m_sph] = ((-1)**m_sph) * np.conjugate(sph_harm(l_sph, m_sph, theta, phi))
    return q_l


def func_to_value(l_sph, func):
    sum_norm2 = 0.0
    for i_j in range(2 * l_sph + 1):
        comp = func[i_j]
        sum_norm2 += np.real(comp * np.conjugate(comp))
    q_value = math.sqrt(sum_norm2 * (4.0 * math.pi) / (2.0 * l_sph + 1.0))
    return round(q_value, 14)


def calc_q_wrapper(args):
    [box_length, neighbor_list_ii,
        i_i, l_list, o_factor, oi_oj, p_list] = args

    comb = [(l_sph, o_f, o_i, o_j, p_weight)
            for l_sph in l_list for o_f in o_factor
            for o_i in oi_oj for o_j in oi_oj for p_weight in p_list]

    q_func_temp = {}
    for l_sph, o_f, o_i, o_j, p_weight in comb:
        name = misc_cy.naming('q', [l_sph, 0, 0, o_f, o_i, o_j, p_weight])

        x_i = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_i, o_f, o_i)

        # neighbor
        q_temp = np.array([0 + 0j for i in range(2 * l_sph + 1)])
        for i_j in neighbor_list_ii:
            x_j = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_j, o_f, o_j)
            delta = misc_cy.calc_delta(x_i, x_j, box_length)
            pol = misc_cy.convert_to_theta_phi(delta)
            q_l = calc_q(l_sph, pol['theta'], pol['phi'])

            for i_k in range(2 * l_sph + 1):
                q_temp[i_k] += q_l[i_k]

        # self director
        if type(p_weight) == type('string') and 'N' in p_weight:
            # p_weight == [ 'N', 'N/2', '2*N' ]
            N = len(neighbor_list_ii)
            p_fact = eval(p_weight)
        else:
            p_fact = p_weight

        oi_list_not_oi = []
        for i_j in oi_oj:
            if i_j != o_i:
                oi_list_not_oi.append(i_j)
        for i_j in oi_list_not_oi:
            x_j = [coord_1d[3 * i_i + i] + direct_1d[3 * i_i + i]
                   for i in range(3)]
            delta = misc_cy.calc_delta(
                x_i, x_j, box_length)
            pol = misc_cy.convert_to_theta_phi(
                delta)
            q_l = calc_q(l_sph, pol['theta'], pol['phi'])

            for i_k in range(2 * l_sph + 1):
                q_temp[i_k] += p_fact * q_l[i_k]

        for i_k in range(2 * l_sph + 1):
            q_temp[i_k] = q_temp[i_k] / (float(len(neighbor_list_ii)) + p_fact)

        q_func_temp[name] = q_temp

    return q_func_temp


@cython.boundscheck(False)
@cython.cdivision(True)
def spherical_order_parameter(coord, direct, box_length, setting, neighbor_list, thread_num):
    # [Q_N]_l_a_b_oi_oj_P
    a_times = setting['ave_times']
    b_times = setting['b_in_Q']
    oi_oj = setting['oi_oj']
    o_factor = setting['o_factor']
    l_list = setting['l_in_Q']
    p_list = setting['p_in_Q']

    # calc spherical function
    # prepare parallel
    ### [box_length, neighbor_list[i_i], i_i, l_list, o_factor, oi_oj, p_list]
    ### [box_length, neighbor_list_ii, i_i, l_list, o_factor, oi_oj, p_list]

    cdef:
        int i_i, i_j, i_k, len_c = len(coord), o_f, o_i, o_j, i, j
        int l_neighbor_list_ii, i_2, l_sph, m_sph, p_weight, l_sph2p1
        int l_q_func_name_old_0
        double dist, theta, phi, inv_l_sph2p1, sum_norm2
        double q_value, PI4 = 4.0*math.acos(-1.0), dl_neighbor_list_ii_1, dpart
        double x_i[3]
        double x_j[3]
        double delta[3]
        vector[double] dbox_length = box_length
        vector[vector[double]] dcoord = coord
        vector[vector[double]] ddirect = direct
        vector[vector[int]] ineighbor_list = neighbor_list

        dvec q_val = dvec()
        cvec q_temp = cvec()
        cvec cpart = cvec()
        cvec_vec q_func_name_old = cvec_vec()
        double complex sphharm, part_temp, cl_neighbor_list_ii_1, I = 1j, comp

    q_func_temp= []
    for i_i in range(len_c):

        l_neighbor_list_ii = len(neighbor_list[i_i])
        comb = [(l_sph, o_f, o_i, o_j, p_weight)
                for l_sph in l_list for o_f in o_factor
                for o_i in oi_oj for o_j in oi_oj for p_weight in p_list]

        res = {}
        for l_sph, o_f, o_i, o_j, p_weight in comb:
            name = misc_cy.naming('q', [l_sph, 0, 0, o_f, o_i, o_j, p_weight])
            l_sph2p1 = 2*l_sph+1

            ### x_i = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_i, o_f, o_i)
            for i in range(3):
                x_i[i] = dcoord[i_i][i] + <double>o_f * <double>o_i * ddirect[i_i][i]

            # neighbor
            ### q_temp = np.array([0 + 0j for i in range(2 * l_sph + 1)])
            q_temp = cvec(l_sph2p1)
            ### for i_j in neighbor_list[i_i]:
            for i_2 in range(l_neighbor_list_ii):
                i_j = ineighbor_list[i_i][i_2]
                ### x_j = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_j, o_f, o_j)
                for i in range(3):
                    x_j[i] = dcoord[i_j][i] + <double>o_f * <double>o_j * ddirect[i_j][i]
                ### delta = misc_cy.calc_delta(x_i, x_j, box_length)
                for i in range(3):
                    delta[i] = x_j[i] - x_i[i]
                    if delta[i] < -dbox_length[i] * 0.5:
                        delta[i] += dbox_length[i]
                    elif delta[i] >= dbox_length[i] * 0.5:
                        delta[i] -= dbox_length[i]
                ### pol = misc_cy.convert_to_theta_phi(delta)
                dist = 0.0
                for i in range(3):
                    dist += delta[i]*delta[i]
                dist = math.sqrt(dist)
                theta = math.acos(delta[2] / dist)
                phi = math.atan2(delta[1], delta[0])

                ### q_l = calc_q(l_sph, pol['theta'], pol['phi'])
                ### for i_k in range(2 * l_sph + 1):
                ###     q_temp[i_k] += q_l[i_k]
                q_temp[0] += sph_harm_cy(l_sph, 0, theta, phi)
                for m_sph in range(1, l_sph + 1):
                    sphharm = sph_harm_cy(l_sph, m_sph, theta, phi)
                    q_temp[m_sph] += sphharm
                    q_temp[l_sph2p1-m_sph] += sphharm

            # self director
            if type(p_weight) == type('string') and 'N' in p_weight:
                # p_weight == [ 'N', 'N/2', '2*N' ]
                N = len(neighbor_list[i_i])
                p_fact = eval(p_weight)
            else:
                p_fact = p_weight

            oi_list_not_oi = []
            for i_j in oi_oj:
                if i_j != o_i:
                    oi_list_not_oi.append(i_j)
            for i_j in oi_list_not_oi:
                ### x_j = [coord_1d[3 * i_i + i] + direct_1d[3 * i_i + i] for i in range(3)]
                for i in range(3):
                    x_j = dcoord[i_i][i] + direct[i_i][i]
                ### delta = misc_cy.calc_delta(x_i, x_j, box_length)
                for i in range(3):
                    delta[i] = x_j[i] - x_i[i]
                    if delta[i] < -dbox_length[i] * 0.5:
                        delta[i] += dbox_length[i]
                    elif delta[i] >= dbox_length[i] * 0.5:
                        delta[i] -= dbox_length[i]
                ### pol = misc_cy.convert_to_theta_phi(delta)
                dist = 0.0
                for i in range(3):
                    dist += delta[i]*delta[i]
                dist = math.sqrt(dist)
                theta = math.acos(delta[2] / dist)
                phi = math.atan2(delta[1], delta[0])

                ### q_l = calc_q(l_sph, pol['theta'], pol['phi'])
                ### for i_k in range(2 * l_sph + 1):
                ###     q_temp[i_k] += p_fact * q_l[i_k]
                q_temp[0] += sph_harm_cy(l_sph, 0, theta, phi)*<double complex>p_fact
                for m_sph in range(1, l_sph + 1):
                    sphharm = sph_harm_cy(l_sph, m_sph, theta, phi)*<double complex>p_fact
                    q_temp[m_sph] += sphharm
                    q_temp[l_sph2p1-m_sph] += sphharm

            for i_k in range(2 * l_sph + 1):
                ### q_temp[i_k] = q_temp[i_k] / (float(len(neighbor_list[i_i])) + p_fact)
                q_temp[i_k] = q_temp[i_k] / (<double>l_neighbor_list_ii + <double>p_fact)

            res[name] = q_temp

        q_func_temp.append(res)

    q_func = misc_cy.data_num_name_to_data_name_num(q_func_temp, len_c)

    # calc function average
    comb = [(l_sph, b_t, o_f, o_i, o_j, p_weight)
            for l_sph in l_list for b_t in range(b_times)
            for o_f in o_factor
            for o_i in oi_oj for o_j in oi_oj for p_weight in p_list]

    for l_sph, b_t, o_f, o_i, o_j, p_weight in comb:
        name = misc_cy.naming('q', [l_sph, 0, b_t+1, o_f, o_i, o_j, p_weight])
        name_old = misc_cy.naming('q', [l_sph, 0, b_t, o_f, o_i, o_j, p_weight])
        l_q_func_name_old_0 = len(q_func[name_old][0])
        q_func_name_old = cvec_vec(len_c,cvec(l_q_func_name_old_0))
        for i in range(len_c):
            for j in range(l_q_func_name_old_0):
                q_func_name_old[i][j] = q_func[name_old][i][j]
        ### q_func[name] = misc_cy.v_neighb_ave(neighbor_list, q_func[name_old])
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
    comb = [(o_f, o_i, o_j, p_weight)
            for o_f in o_factor for o_i in oi_oj for o_j in oi_oj for p_weight in p_list]
    op_data = {}
    for l_sph in l_list:
        l_sph2p1 = 2*l_sph+1
        inv_l_sph2p1 = 1.0/<double>l_sph2p1
        for b_t in range(b_times + 1):
            for o_f, o_i, o_j, p_weight in comb:
                name = misc_cy.naming('q', [l_sph, 0, b_t, o_f, o_i, o_j, p_weight])
                ### q_val = []
                q_val = dvec()
                for i_i in range(len_c):
                    ### q_val.append(func_to_value(l_sph, q_func[name][i_i]))
                    ### def func_to_value(l_sph, func):
                    sum_norm2 = 0.0
                    for i_j in range(l_sph2p1):
                        comp = q_func[name][i_i][i_j]
                        ### sum_norm2 += np.real(comp * np.conjugate(comp))
                        sum_norm2 += (comp*comp.conjugate()).real
                    q_value = math.sqrt(sum_norm2*PI4*inv_l_sph2p1)
                    q_value = round(q_value, 14)

                    ### q_val.push_back(func_to_value(l_sph, q_func[name][i_i]))
                    q_val.push_back(q_value)
                op_data[name] = q_val

    # neighbor value averaging
    comb = [(l_sph, a_t, b_t, o_f, o_i, o_j, p_weight)
            for l_sph in l_list
            for a_t in range(a_times)
            for b_t in range(b_times+1)
            for o_f in o_factor for o_i in oi_oj for o_j in oi_oj for p_weight in p_list]
    for l_sph, a_t, b_t, o_f, o_i, o_j, p_weight in comb:
        name     = misc_cy.naming('q', [l_sph, a_t+1, b_t, o_f, o_i, o_j, p_weight])
        name_old = misc_cy.naming('q', [l_sph, a_t  , b_t, o_f, o_i, o_j, p_weight])
        ### op_data[name] = misc_cy.v_neighb_ave(neighbor_list, op_data[name_old])
        ### val_ave = []
        q_val = dvec()
        for i_i in range(len_c):
            l_neighbor_list_ii = len(neighbor_list[i_i])
            dl_neighbor_list_ii_1 = <double>(l_neighbor_list_ii + 1)
            dpart = 0.0
            for ii_j in range(l_neighbor_list_ii):
                inei = ineighbor_list[i_i][ii_j]
                dpart += op_data[name_old][inei]
                ### part.append(op_data[name_old][inei])
            dpart += op_data[name_old][i_i]
            ### val_ave.append(np.average(part))
            q_val.push_back(dpart/dl_neighbor_list_ii_1)
        ### op_data[name] = val_ave
        op_data[name] = q_val

    return op_data

def spherical_order_parameter_org(coord, direct, box_length, setting, neighbor_list, thread_num):
    # [Q_N]_l_a_b_oi_oj_P
    a_times = setting['ave_times']
    b_times = setting['b_in_Q']
    oi_oj = setting['oi_oj']
    o_factor = setting['o_factor']
    l_list = setting['l_in_Q']
    p_list = setting['p_in_Q']

    # calc spherical function
    # prepare parallel
    global coord_1d
    coord_1d = Array('d', misc_cy.convert_3dim_to_1dim(coord), lock=False)

    global direct_1d
    direct_1d = Array('d', misc_cy.convert_3dim_to_1dim(direct), lock=False)

    now_pool = Pool(thread_num)
    args = [[box_length, neighbor_list[i_i], i_i, l_list, o_factor, oi_oj, p_list]
            for i_i in range(len(coord))]
    q_func_temp = now_pool.map(calc_q_wrapper, args)
    now_pool.close()

    del coord_1d
    del direct_1d

    q_func = misc_cy.data_num_name_to_data_name_num(q_func_temp, len(coord))

    # calc function average
    comb = [(l_sph, b_t, o_f, o_i, o_j, p_weight)
            for l_sph in l_list for b_t in range(b_times)
            for o_f in o_factor
            for o_i in oi_oj for o_j in oi_oj for p_weight in p_list]

    for l_sph, b_t, o_f, o_i, o_j, p_weight in comb:
        name = misc_cy.naming('q', [l_sph, 0, b_t+1, o_f, o_i, o_j, p_weight])
        name_old = misc_cy.naming('q', [l_sph, 0, b_t, o_f, o_i, o_j, p_weight])
        q_func[name] = misc_cy.v_neighb_ave(neighbor_list, q_func[name_old])

    # func to value
    comb = [(o_f, o_i, o_j, p_weight)
            for o_f in o_factor for o_i in oi_oj for o_j in oi_oj for p_weight in p_list]
    op_data = {}
    for l_sph in l_list:
        for b_t in range(b_times + 1):
            for o_f, o_i, o_j, p_weight in comb:
                name = misc_cy.naming('q', [l_sph, 0, b_t, o_f, o_i, o_j, p_weight])
                q_val = []
                for i_i in range(len(coord)):
                    q_val.append(func_to_value(l_sph, q_func[name][i_i]))
                op_data[name] = q_val

    # neighbor value averaging
    comb = [(l_sph, a_t, b_t, o_f, o_i, o_j, p_weight)
            for l_sph in l_list
            for a_t in range(a_times)
            for b_t in range(b_times+1)
            for o_f in o_factor for o_i in oi_oj for o_j in oi_oj for p_weight in p_list]
    for l_sph, a_t, b_t, o_f, o_i, o_j, p_weight in comb:
        name     = misc_cy.naming('q' , [l_sph , a_t+1 , b_t, o_f , o_i , o_j , p_weight])
        name_old = misc_cy.naming('q' , [l_sph , a_t   , b_t, o_f , o_i , o_j , p_weight])
        op_data[name] = misc_cy.v_neighb_ave(neighbor_list, op_data[name_old])

    return op_data
