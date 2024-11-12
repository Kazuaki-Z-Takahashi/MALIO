# -*- coding: utf-8 -*-

import numpy as np
from multiprocessing import Pool, Array
import misc_cy
cimport cython
from malio_cy_def cimport *

@cython.boundscheck(False)
@cython.cdivision(True)
def nda_order_parameter(coord, direct, box_length, setting, neighbor_list, thread_num):
    a_times = setting['ave_times']
    o_factor = setting['o_factor']
    oi_oj_ok = setting['oi_oj_ok']
    func = setting['func']

    ### coord_1d = misc_cy.convert_3dim_to_1dim(coord)
    ### direct_1d = misc_cy.convert_3dim_to_1dim(direct)
    ### [box_length, neighbor_list_ii,i_i, o_factor, oi_oj_ok, func]
    ### [box_length, neighbor_list[i_i], i_i, o_factor, oi_oj_ok, func]

    cdef:
        int i_i, i_j, i_k, len_c = len(coord), o_f, o_i, o_j, o_k, i, j
        int N, i_2, i_3, l_func
        double d_sum, di_j
        double x_i[3]
        double x_j[3]
        double x_k[3]
        double x_i_j[3]
        double x_ik[3]
        double xjk[3]
        vector[double] dbox_length = box_length
        vector[vector[double]] dcoord = coord
        vector[vector[double]] ddirect = direct
        vector[vector[int]] ineighbor_list = neighbor_list

    op_val_temp= []
    l_func = len(func)
    for i_i in range(len_c):

        N = len(neighbor_list[i_i])
        comb = [(o_f, o_i, o_j, o_k, f_1, f_2, f_3)
            for o_f in o_factor for o_i in oi_oj_ok for o_j in oi_oj_ok for o_k in oi_oj_ok
            for f_1 in range(l_func)
            for f_2 in range(l_func)
            for f_3 in range(l_func)]

        op_temp = {}
        for o_f, o_i, o_j, o_k, f_1, f_2, f_3 in comb:
            name = misc_cy.naming('d', [0, o_f, o_i, o_j, o_k, f_1, f_2, f_3])
            ### x_i = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_i, o_f, o_i)
            for i in range(3):
                x_i[i] = dcoord[i_i][i] + <double>o_f * <double>o_i * ddirect[i_i][i]

            d_sum = 0.0
            for i_2 in range(N - 1):
                i_j = ineighbor_list[i_i][i_2]
                ### x_j = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_j, o_f, o_j)
                for i in range(3):
                    x_j[i] = dcoord[i_j][i] + <double>o_f * <double>o_j * ddirect[i_j][i]
                ### x_i_j = misc_cy.calc_delta(x_j, x_i, box_length)
                for i in range(3):
                    x_i_j[i] = x_j[i] - x_i[i]
                    if x_i_j[i] < -dbox_length[i] * 0.5:
                        x_i_j[i] += dbox_length[i]
                    elif x_i_j[i] >= dbox_length[i] * 0.5:
                        x_i_j[i] -= dbox_length[i]
                ### di_j = np.linalg.norm(x_i_j)
                di_j = 0.0
                for i in range(3):
                    di_j += x_i_j[i]**2
                di_j = math.sqrt(di_j)

                for i3 in range(i_2 + 1, N):
                    i_k = ineighbor_list[i_i][i3]
                    ### x_k = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_k, o_f, o_k)
                    for i in range(3):
                        x_k[i] = dcoord[i_k][i] + <double>o_f * <double>o_k * ddirect[i_k][i]
                    ### x_ik = misc_cy.calc_delta(x_k, x_i, box_length)
                    ### xjk = misc_cy.calc_delta(x_k, x_j, box_length)
                    for i in range(3):
                        x_ik[i] = x_k[i] - x_i[i]
                        xjk[i] = x_k[i] - x_j[i]
                        if x_ik[i] < -dbox_length[i] * 0.5:
                            x_ik[i] += dbox_length[i]
                        elif x_ik[i] >= dbox_length[i] * 0.5:
                            x_ik[i] -= dbox_length[i]
                        if xjk[i] < -dbox_length[i] * 0.5:
                            xjk[i] += dbox_length[i]
                        elif xjk[i] >= dbox_length[i] * 0.5:
                            xjk[i] -= dbox_length[i]
                    ### dik = np.linalg.norm(x_ik)
                    ### djk = np.linalg.norm(xjk)
                    dik = 0.0
                    djk = 0.0
                    for i in range(3):
                        dik += x_ik[i]**2
                        djk += xjk[i]**2
                    dik = math.sqrt(dik)
                    djk = math.sqrt(djk)

                    d_sum += func[f_1](di_j) * func[f_2](dik) * func[f_3](djk)
            ### op_temp[name] = d_sum / (N * (N - 1) / 2)
            op_temp[name] = d_sum / (<double>(N*N-N)*0.5)

        op_val_temp.append(op_temp)

    op_value = misc_cy.data_num_name_to_data_name_num(op_val_temp, len_c)

    # neighbor value averaging
    comb = [(o_f, o_i, o_j, o_k, f_1, f_2, f_3)
            for o_f in o_factor
            for o_i in oi_oj_ok for o_j in oi_oj_ok for o_k in oi_oj_ok
            for f_1 in range(len(func))
            for f_2 in range(len(func))
            for f_3 in range(len(func))]
    for a_t in range(a_times):
        for o_f, o_i, o_j, o_k, f_1, f_2, f_3 in comb:
            name = misc_cy.naming('d', [a_t+1, o_f, o_i, o_j, o_k, f_1, f_2, f_3])
            name_old = misc_cy.naming(
                'd', [a_t, o_f, o_i, o_j, o_k, f_1, f_2, f_3])
            op_value[name] = misc_cy.v_neighb_ave(
                neighbor_list, op_value[name_old])

    return op_value


def calc_d_wrapper(args):
    [box_length, neighbor_list_ii,
        i_i, o_factor, oi_oj_ok, func] = args

    comb = [(o_f, o_i, o_j, o_k, f_1, f_2, f_3)
            for o_f in o_factor for o_i in oi_oj_ok for o_j in oi_oj_ok for o_k in oi_oj_ok
            for f_1 in range(len(func))
            for f_2 in range(len(func))
            for f_3 in range(len(func))]

    op_temp = {}
    for o_f, o_i, o_j, o_k, f_1, f_2, f_3 in comb:
        name = misc_cy.naming('d', [0, o_f, o_i, o_j, o_k, f_1, f_2, f_3])

        x_i = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_i, o_f, o_i)

        N = len(neighbor_list_ii)
        d_sum = 0.0
        for i_2 in range(N - 1):
            i_j = neighbor_list_ii[i_2]
            x_j = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_j, o_f, o_j)
            x_i_j = misc_cy.calc_delta(x_j, x_i, box_length)
            di_j = np.linalg.norm(x_i_j)

            for i3 in range(i_2 + 1, N):
                i_k = neighbor_list_ii[i3]
                x_k = misc_cy.calc_head_coordinate(
                    coord_1d, direct_1d, i_k, o_f, o_k)
                x_ik = misc_cy.calc_delta(x_k, x_i, box_length)
                xjk = misc_cy.calc_delta(x_k, x_j, box_length)
                dik = np.linalg.norm(x_ik)
                djk = np.linalg.norm(xjk)

                d_sum += func[f_1](di_j) * func[f_2](dik) * func[f_3](djk)
        op_temp[name] = d_sum / (N * (N - 1) / 2)

    return op_temp


def nda_order_parameter_org(coord, direct, box_length, setting, neighbor_list, thread_num):
    a_times = setting['ave_times']
    o_factor = setting['o_factor']
    oi_oj_ok = setting['oi_oj_ok']
    func = setting['func']

    # prepare parallel
    global coord_1d
    coord_1d = Array('d', misc_cy.convert_3dim_to_1dim(coord), lock=False)

    global direct_1d
    direct_1d = Array('d', misc_cy.convert_3dim_to_1dim(direct), lock=False)

    now_pool = Pool(thread_num)
    args = [[box_length, neighbor_list[i_i],
             i_i, o_factor, oi_oj_ok, func]
            for i_i in range(len(coord))]
    op_val_temp = now_pool.map(calc_d_wrapper, args)
    now_pool.close()

    del coord_1d
    del direct_1d

    op_value = misc_cy.data_num_name_to_data_name_num(op_val_temp, len(coord))

    # neighbor value averaging
    comb = [(o_f, o_i, o_j, o_k, f_1, f_2, f_3)
            for o_f in o_factor
            for o_i in oi_oj_ok for o_j in oi_oj_ok for o_k in oi_oj_ok
            for f_1 in range(len(func))
            for f_2 in range(len(func))
            for f_3 in range(len(func))]
    for a_t in range(a_times):
        for o_f, o_i, o_j, o_k, f_1, f_2, f_3 in comb:
            name = misc_cy.naming('d', [a_t+1, o_f, o_i, o_j, o_k, f_1, f_2, f_3])
            name_old = misc_cy.naming(
                'd', [a_t, o_f, o_i, o_j, o_k, f_1, f_2, f_3])
            op_value[name] = misc_cy.v_neighb_ave(
                neighbor_list, op_value[name_old])

    return op_value
