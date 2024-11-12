# -*- coding: utf-8 -*-

### import math
from multiprocessing import Pool, Array
import misc_cy
cimport cython
from malio_cy_def cimport *

@cython.boundscheck(False)
@cython.cdivision(True)
def baa_order_parameter(coord, direct, box_length, setting, neighbor_list, thread_num):
    a_times = setting['ave_times']
    m_l = setting['m']
    phi_l = setting['phi']
    n_l = setting['n']
    o_factor = setting['o_factor']
    oi_oj_ok = setting['oi_oj_ok']

    cdef:
        int i_i, i_j, i_k, len_c = len(coord), o_f, o_i, o_j, o_k, i, j
        int l_neighbor_list_i_i, i_2, i3, n_pow, m_fac
        double n_1, op_temp, theta, phi, pdot, nrm1, nrm2
        double x_i[3]
        double x_j[3]
        double x_k[3]
        double x_i_j[3]
        double x_ik[3]
        double sum_vec[3]
        double sum_vec_m[3]
        vector[double] dbox_length = box_length
        vector[vector[double]] dcoord = coord
        vector[vector[double]] ddirect = direct
        vector[vector[int]] ineighbor_list = neighbor_list

    op_val_temp= []
    for i_i in range(len_c):
        l_neighbor_list_i_i = len(neighbor_list[i_i])

        comb = [(m_fac, phi, n_pow, o_f, o_i, o_j, o_k)
                for m_fac in m_l for phi in phi_l for n_pow in n_l
                for o_f in o_factor for o_i in oi_oj_ok for o_j in oi_oj_ok
                for o_k in oi_oj_ok]
        b_list = {}
        for m_fac, phi, n_pow, o_f, o_i, o_j, o_k in comb:
            ### x_i = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_i, o_f, o_i)
            ### x_i = []
            for i in range(3):
                x_i[i] = dcoord[i_i][i] + <double>o_f * <double>o_i * ddirect[i_i][i]
                ### x_i.append(coord[i_i][i] + o_f * o_i * direct[i_i][i])
            op_temp = 0.0
            for i_2 in range(l_neighbor_list_i_i - 1):
                i_j = ineighbor_list[i_i][i_2]
                ### x_j = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_j, o_f, o_j)
                ### x_j = []
                for i in range(3):
                    x_j[i] = dcoord[i_j][i] + <double>o_f * <double>o_j * ddirect[i_j][i]
                    ### x_j.append(coord[i_j][i] + o_f * o_j * direct[i_j][i])
                for i3 in range(i_2 + 1, l_neighbor_list_i_i):
                    i_k = ineighbor_list[i_i][i3]
                    ### x_k = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_k, o_f, o_k)
                    ### x_k = []
                    for i in range(3):
                        x_k[i] = dcoord[i_k][i] + <double>o_f * <double>o_k * ddirect[i_k][i]
                        ### x_k.append(coord[i_k][i] + o_f * o_k * direct[i_k][i])
                    ### x_i_j = misc_cy.calc_delta(x_j, x_i, box_length)
                    ### x_ik  = misc_cy.calc_delta(x_k, x_i, box_length)
                    for i in range(3):
                        x_i_j[i] = x_j[i] - x_i[i]
                        x_ik[i]  = x_k[i] - x_i[i]
                        if x_i_j[i] < -dbox_length[i] * 0.5:
                            x_i_j[i] += dbox_length[i]
                        elif x_i_j[i] >= dbox_length[i] * 0.5:
                            x_i_j[i] -= dbox_length[i]
                        if x_ik[i] < -dbox_length[i] * 0.5:
                            x_ik[i] += dbox_length[i]
                        elif x_ik[i] >= dbox_length[i] * 0.5:
                            x_ik[i] -= dbox_length[i]
                    pdot = 0.0
                    nrm1 = 0.0
                    nrm2 = 0.0
                    for i in range(3):
                        pdot += x_i_j[i]*x_ik[i]
                        nrm1 += x_i_j[i]*x_i_j[i]
                        nrm2 += x_ik[i]*x_ik[i]
                    if math.sqrt(nrm1*nrm2) > 1.0e-10:
                        theta = math.acos(pdot/math.sqrt(nrm1*nrm2))
                    else:
                        theta = 0.0
                    ### try:
                    ###     theta = misc_cy.angle(x_i_j, x_ik)
                    ### except ValueError:
                    ###     theta = 0.0

                    op_temp += math.pow((math.cos(<double>m_fac * theta + phi)),n_pow)
            n_1 = <double>l_neighbor_list_i_i
            op_temp = op_temp / ((n_1*n_1 - n_1)*0.5)

            name = misc_cy.naming('b', [0, m_fac, phi, n_pow, o_f, o_i, o_j, o_k])
            b_list[name] = op_temp
        op_val_temp.append(b_list)

    op_value = misc_cy.data_num_name_to_data_name_num(op_val_temp, len_c)

    # neighbor value averaging
    comb = [(m_fac, phi, n_pow, o_f, o_i, o_j, o_k)
            for m_fac in m_l for phi in phi_l for n_pow in n_l
            for o_f in o_factor for o_i in oi_oj_ok for o_j in oi_oj_ok
            for o_k in oi_oj_ok]

    for a_t in range(a_times):
        for m_fac, phi, n_pow, o_f, o_i, o_j, o_k in comb:
            name = misc_cy.naming('b', [a_t+1, m_fac, phi, n_pow, o_f, o_i, o_j, o_k])
            name_old = misc_cy.naming('b', [a_t, m_fac, phi, n_pow, o_f, o_i, o_j, o_k])
            op_value[name] = misc_cy.v_neighb_ave(neighbor_list, op_value[name_old])

    return op_value

def calc_baa_wrapper(args):
    [box_length, neighbor_list_ii, i_i,
        m_l, phi_l, n_l, o_factor, oi_oj_ok] = args

    comb = [(m_fac, phi, n_pow, o_f, o_i, o_j, o_k)
            for m_fac in m_l for phi in phi_l for n_pow in n_l
            for o_f in o_factor for o_i in oi_oj_ok for o_j in oi_oj_ok
            for o_k in oi_oj_ok]

    b_list = {}
    for m_fac, phi, n_pow, o_f, o_i, o_j, o_k in comb:
        x_i = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_i, o_f, o_i)
        op_temp = 0
        for i_2 in range(len(neighbor_list_ii) - 1):
            i_j = neighbor_list_ii[i_2]
            x_j = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_j, o_f, o_j)
            for i3 in range(i_2 + 1, len(neighbor_list_ii)):
                i_k = neighbor_list_ii[i3]
                x_k = misc_cy.calc_head_coordinate(
                    coord_1d, direct_1d, i_k, o_f, o_k)

                x_i_j = misc_cy.calc_delta(x_j, x_i, box_length)
                x_ik = misc_cy.calc_delta(x_k, x_i, box_length)
                try:
                    theta = misc_cy.angle(x_i_j, x_ik)
                except ValueError:
                    theta = 0.0

                op_temp += (math.cos(m_fac * theta + phi))**n_pow
        n_1 = float(len(neighbor_list_ii))
        op_temp = op_temp / (n_1 * (n_1 - 1) / 2)

        name = misc_cy.naming('b', [0, m_fac, phi, n_pow, o_f, o_i, o_j, o_k])
        b_list[name] = op_temp

    return b_list


def baa_order_parameter_org(coord, direct, box_length, setting, neighbor_list, thread_num):
    a_times = setting['ave_times']
    m_l = setting['m']
    phi_l = setting['phi']
    n_l = setting['n']
    o_factor = setting['o_factor']
    oi_oj_ok = setting['oi_oj_ok']

    # prepare parallel
    global coord_1d
    coord_1d = Array('d', misc_cy.convert_3dim_to_1dim(coord), lock=False)

    global direct_1d
    direct_1d = Array('d', misc_cy.convert_3dim_to_1dim(direct), lock=False)

    # calc b
    now_pool = Pool(thread_num)
    args = [[box_length, neighbor_list[i_i], i_i, m_l, phi_l, n_l, o_factor, oi_oj_ok]
            for i_i in range(len(coord))]
    op_val_temp = now_pool.map(calc_baa_wrapper, args)
    now_pool.close()

    del coord_1d
    del direct_1d

    op_value = misc_cy.data_num_name_to_data_name_num(op_val_temp, len(coord))

    # neighbor value averaging
    comb = [(m_fac, phi, n_pow, o_f, o_i, o_j, o_k)
            for m_fac in m_l for phi in phi_l for n_pow in n_l
            for o_f in o_factor for o_i in oi_oj_ok for o_j in oi_oj_ok
            for o_k in oi_oj_ok]

    for a_t in range(a_times):
        for m_fac, phi, n_pow, o_f, o_i, o_j, o_k in comb:
            name = misc_cy.naming('b', [a_t+1, m_fac, phi, n_pow, o_f, o_i, o_j, o_k])
            name_old = misc_cy.naming('b', [a_t, m_fac, phi, n_pow, o_f, o_i, o_j, o_k])
            op_value[name] = misc_cy.v_neighb_ave(neighbor_list, op_value[name_old])

    return op_value
