# -*- coding: utf-8 -*-

### import math
from multiprocessing import Pool, Array
import numpy as np
import misc_cy
cimport cython
from malio_cy_def cimport *

@cython.boundscheck(False)
@cython.cdivision(True)
def afs_order_parameter(coord, direct, box_length, setting, neighbor_list, thread_num):
    a_times = setting['ave_times']
    o_factor = setting['o_factor']
    oi_oj_ok = setting['oi_oj_ok']
    l_list = setting['l_in_F']
    func = setting['func']

    ### coord_1d = misc_cy.convert_3dim_to_1dim(coord)
    ### direct_1d = misc_cy.convert_3dim_to_1dim(direct)

    cdef:
        int i_i, i_j, i_k, len_c = len(coord), o_f, o_i, o_j, o_k, i, j
        int N, i_2, i_3, l_func, l
        double d_sum, di_j, theta, pdot, nrm1, nrm2
        double x_i[3]
        double x_j[3]
        double x_k[3]
        double x_i_j[3]
        double x_ik[3]
        vector[double] dbox_length = box_length
        vector[vector[double]] dcoord = coord
        vector[vector[double]] ddirect = direct
        vector[vector[int]] ineighbor_list = neighbor_list

    op_val_temp= []
    l_func = len(func)
    for i_i in range(len(coord)):
        N = len(neighbor_list[i_i])
        comb = [(o_f, o_i, o_j, o_k, f_1, f_2, l)
                for o_f in o_factor for o_i in oi_oj_ok for o_j in oi_oj_ok for o_k in oi_oj_ok
                for f_1 in range(l_func)
                for f_2 in range(l_func)
                for l in l_list]

        op_temp = {}
        for o_f, o_i, o_j, o_k, f_1, f_2, l in comb:
            name = misc_cy.naming('f', [0, o_f, o_i, o_j, o_k, f_1, f_2, l])

            ### x_i = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_i, o_f, o_i)
            for i in range(3):
                x_i[i] = dcoord[i_i][i] + <double>o_f * <double>o_i * ddirect[i_i][i]

            d_sum = 0.0
            for i_2 in range(N - 1):
                i_j = ineighbor_list[i_i][i_2]
                ### x_j = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_j, o_f, o_j)
                for i in range(3):
                    x_j[i] = dcoord[i_j][i] + <double>o_f * <double>o_j * ddirect[i_j][i]
                ### x_i_j = misc_cy.calc_delta(x_j, x_i, dbox_length)
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
                    ### x_ik = misc_cy.calc_delta(x_k, x_i, dbox_length)
                    for i in range(3):
                        x_ik[i] = x_k[i] - x_i[i]
                        if x_ik[i] < -dbox_length[i] * 0.5:
                            x_ik[i] += dbox_length[i]
                        elif x_ik[i] >= dbox_length[i] * 0.5:
                            x_ik[i] -= dbox_length[i]
                    ### di_k = np.linalg.norm(x_ik)
                    di_k = 0.0
                    for i in range(3):
                        di_k += x_ik[i]**2
                    di_k = math.sqrt(di_k)

                    ### try:
                    ###     theta = misc_cy.angle(x_i_j, x_ik)
                    ### except ValueError:
                    ###     theta = 0.0
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
                    if theta >= math.pi:
                        theta -= math.pi

                    d_sum += func[f_1](di_j) * func[f_2](di_k) * math.cos(<double>l*theta)
            if N <= 1:
                op_temp[name] = 0.0
            else:
                ### op_temp[name] = d_sum / (N * (N - 1) / 2)
                op_temp[name] = d_sum / (<double>(N*N-N)*0.5)

        op_val_temp.append(op_temp)

    op_value = misc_cy.data_num_name_to_data_name_num(op_val_temp, len(coord))

    # neighbor value averaging
    comb = [(o_f, o_i, o_j, o_k, f_1, f_2, l)
            for o_f in o_factor
            for o_i in oi_oj_ok for o_j in oi_oj_ok for o_k in oi_oj_ok
            for f_1 in range(len(func))
            for f_2 in range(len(func))
            for l in l_list]
    for a_t in range(a_times):
        for o_f, o_i, o_j, o_k, f_1, f_2, l in comb:
            name = misc_cy.naming('f', [a_t+1, o_f, o_i, o_j, o_k, f_1, f_2, l])
            name_old = misc_cy.naming( 'f', [a_t, o_f, o_i, o_j, o_k, f_1, f_2, l])
            op_value[name] = misc_cy.v_neighb_ave(
                neighbor_list, op_value[name_old])

    return op_value

def calc_f_wrapper(args):
    [box_length, neighbor_list_ii,
        i_i, o_factor, oi_oj_ok, func, l_list] = args

    comb = [(o_f, o_i, o_j, o_k, f_1, f_2, l)
            for o_f in o_factor for o_i in oi_oj_ok for o_j in oi_oj_ok for o_k in oi_oj_ok
            for f_1 in range(len(func))
            for f_2 in range(len(func))
            for l in l_list]

    op_temp = {}
    for o_f, o_i, o_j, o_k, f_1, f_2, l in comb:
        name = misc_cy.naming('f', [0, o_f, o_i, o_j, o_k, f_1, f_2, l])

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
                di_k = np.linalg.norm(x_ik)

                try:
                    theta = misc_cy.angle(x_i_j, x_ik)
                except ValueError:
                    theta = 0.0
                if theta >= math.pi:
                    theta -= math.pi

                d_sum += func[f_1](di_j) * func[f_2](di_k) * math.cos(l*theta)
        if N <= 1:
            op_temp[name] = 0
        else:
            op_temp[name] = d_sum / (N * (N - 1) / 2)

    return op_temp


def afs_order_parameter_org(coord, direct, box_length, setting, neighbor_list, thread_num):
    a_times = setting['ave_times']
    o_factor = setting['o_factor']
    oi_oj_ok = setting['oi_oj_ok']
    l_list = setting['l_in_F']
    func = setting['func']

    # prepare parallel
    global coord_1d
    coord_1d = Array('d', misc_cy.convert_3dim_to_1dim(coord), lock=False)

    global direct_1d
    direct_1d = Array('d', misc_cy.convert_3dim_to_1dim(direct), lock=False)

    now_pool = Pool(thread_num)
    args = [[box_length, neighbor_list[i_i],
             i_i, o_factor, oi_oj_ok, func, l_list]
            for i_i in range(len(coord))]
    op_val_temp = now_pool.map(calc_f_wrapper, args)
    now_pool.close()

    del coord_1d
    del direct_1d

    op_value = misc_cy.data_num_name_to_data_name_num(op_val_temp, len(coord))

    # neighbor value averaging
    comb = [(o_f, o_i, o_j, o_k, f_1, f_2, l)
            for o_f in o_factor
            for o_i in oi_oj_ok for o_j in oi_oj_ok for o_k in oi_oj_ok
            for f_1 in range(len(func))
            for f_2 in range(len(func))
            for l in l_list]
    for a_t in range(a_times):
        for o_f, o_i, o_j, o_k, f_1, f_2, l in comb:
            name = misc_cy.naming('f', [a_t+1, o_f, o_i, o_j, o_k, f_1, f_2, l])
            name_old = misc_cy.naming( 'f', [a_t, o_f, o_i, o_j, o_k, f_1, f_2, l])
            op_value[name] = misc_cy.v_neighb_ave(
                neighbor_list, op_value[name_old])

    return op_value
