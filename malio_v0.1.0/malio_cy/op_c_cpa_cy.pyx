# -*- coding: utf-8 -*-

from multiprocessing import Pool, Array
import numpy as np
import misc_cy
cimport cython
from malio_cy_def cimport *

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef cpa_order_parameter(coord, direct, box_length, setting, neighbor_list, thread_num):
    a_times = setting['ave_times']
    o_factor = setting['o_factor']
    oi_oj_ok = setting['oi_oj_ok']

    ### coord_1d = misc_cy.convert_3dim_to_1dim(coord)
    ### direct_1d = misc_cy.convert_3dim_to_1dim(direct)
    ### [box_length, neighbor_list[i_i], i_i, o_factor, oi_oj_ok]
    ### [box_length, neighbor_list_ii, i_i, o_factor, oi_oj_ok]

    cdef:
        int i_i, i_j, i_k, len_c = len(coord), o_f, o_i, o_j, o_k, i, j
        int l_neighbor_list_i_i, i_2, half_N, nearest_i_k, i_kk, ii_kk
        double sum_dist, nearest_distnace, distance
        double x_i[3]
        double x_j[3]
        double x_k[3]
        double x_i_j[3]
        double x_i_k[3]
        double x_j_opposite[3]
        double x_j_o_k[3]
        vector[double] dbox_length = box_length
        vector[vector[double]] dcoord = coord
        vector[vector[double]] ddirect = direct
        vector[vector[int]] ineighbor_list = neighbor_list

    op_val_temp= []
    for i_i in range(len_c):
        comb = [(o_f, o_i, o_j, o_k)
            for o_f in o_factor for o_i in oi_oj_ok for o_j in oi_oj_ok for o_k in oi_oj_ok]
        l_neighbor_list_i_i = len(neighbor_list[i_i])

        op_temp = {}
        for o_f, o_i, o_j, o_k in comb:
            name = misc_cy.naming('c', [0, o_f, o_i, o_j, o_k])
            ### x_i = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_i, o_f, o_i)
            for i in range(3):
                x_i[i] = dcoord[i_i][i] + <double>o_f * <double>o_i * ddirect[i_i][i]
            half_N = <int>(<double>l_neighbor_list_i_i*0.5)
            sum_dist = 0.0
            for i_2 in range(half_N):
                i_j = ineighbor_list[i_i][i_2]

                ### x_j = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_j, o_f, o_j)
                for i in range(3):
                    x_j[i] = dcoord[i_j][i] + <double>o_f * <double>o_j * ddirect[i_j][i]

                ### i_k = misc_cy.search_opposite_j_particle(coord_1d, direct_1d, neighbor_list[i_i],
                ###                                     x_i, i_j, x_j, box_length, o_f, o_k)
                ### x_i_j = calc_delta(x_j, x_i, box_length)
                for i in range(3):
                    x_i_j[i] = x_j[i] - x_i[i]
                    if x_i_j[i] < -dbox_length[i] * 0.5:
                        x_i_j[i] += dbox_length[i]
                    elif x_i_j[i] >= dbox_length[i] * 0.5:
                        x_i_j[i] -= dbox_length[i]
                ### x_j_opposite = [x_i[i] - x_i_j[i] for i in range(3)]
                for i in range(3):
                    x_j_opposite[i] = x_i[i] - x_i_j[i]
                nearest_i_k = 1000
                nearest_distnace = 10000000.0
                ### for i_k in neighbor_list_ii:
                for i_kk in range(l_neighbor_list_i_i):
                    ii_kk = ineighbor_list[i_i][i_kk]
                    if ii_kk == i_j:
                        continue
                    ### x_k = calc_head_coordinate(coord_1d, direct_1d, i_k, o_f, o_k)
                    for i in range(3):
                        x_k[i] = dcoord[ii_kk][i] + <double>o_f * <double>o_k * ddirect[ii_kk][i]
                    ### x_j_o_k = calc_delta(x_k, x_j_opposite, box_length)
                    for i in range(3):
                        x_j_o_k[i] = x_k[i] - x_j_opposite[i]
                        if x_j_o_k[i] < -dbox_length[i] * 0.5:
                            x_j_o_k[i] += dbox_length[i]
                        elif x_j_o_k[i] >= dbox_length[i] * 0.5:
                            x_j_o_k[i] -= dbox_length[i]
                    ### distance = np.linalg.norm(x_j_o_k)
                    distance = 0.0
                    for i in range(3):
                        distance += x_j_o_k[i]**2
                    distance = math.sqrt(distance)
                    if distance <= nearest_distnace:
                        nearest_distnace = distance
                        nearest_i_k = ii_kk
                i_k = nearest_i_k

                ### x_k = misc_cy.calc_head_coordinate( coord_1d, direct_1d, i_k, o_f, o_k)
                for i in range(3):
                    x_k[i] = dcoord[i_k][i] + <double>o_f * <double>o_k * ddirect[i_k][i]
                ### x_i_j = misc_cy.calc_delta(x_j, x_i, box_length)
                ### x_i_k = misc_cy.calc_delta(x_k, x_i, box_length)
                for i in range(3):
                        x_i_j[i] = x_j[i] - x_i[i]
                        x_i_k[i] = x_k[i] - x_i[i]
                        if x_i_j[i] < -dbox_length[i] * 0.5:
                            x_i_j[i] += dbox_length[i]
                        elif x_i_j[i] >= dbox_length[i] * 0.5:
                            x_i_j[i] -= dbox_length[i]
                        if x_i_k[i] < -dbox_length[i] * 0.5:
                            x_i_k[i] += dbox_length[i]
                        elif x_i_k[i] >= dbox_length[i] * 0.5:
                            x_i_k[i] -= dbox_length[i]
                ### x_ij_ik = [x_i_j[i] + x_i_k[i] for i in range(3)]
                ### sum_dist += np.dot(x_ij_ik, x_ij_ik)
                for i in range(3):
                    sum_dist += (x_i_j[i] + x_i_k[i])**2

            op_temp[name] = sum_dist / <double>half_N
        op_val_temp.append(op_temp)

    op_value = misc_cy.data_num_name_to_data_name_num(op_val_temp, len_c)
    comb = [(o_f, o_i, o_j, o_k)
            for o_f in o_factor for o_i in oi_oj_ok for o_j in oi_oj_ok for o_k in oi_oj_ok]

    # neighbor value averaging
    for a_t in range(a_times):
        for o_f, o_i, o_j, o_k in comb:
            name = misc_cy.naming('c', [a_t+1, o_f, o_i, o_j, o_k])
            name_old = misc_cy.naming('c', [a_t, o_f, o_i, o_j, o_k])
            op_value[name] = misc_cy.v_neighb_ave(
                neighbor_list, op_value[name_old])

    return op_value

def calc_c_wrapper(args):
    [box_length, neighbor_list_ii,
        i_i, o_factor, oi_oj_ok] = args

    comb = [(o_f, o_i, o_j, o_k)
            for o_f in o_factor for o_i in oi_oj_ok for o_j in oi_oj_ok for o_k in oi_oj_ok]

    nei_ii = neighbor_list_ii
    op_temp = {}
    for o_f, o_i, o_j, o_k in comb:
        name = misc_cy.naming('c', [0, o_f, o_i, o_j, o_k])
        x_i = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_i, o_f, o_i)

        half_N = int(len(nei_ii) / 2)
        sum_dist = 0
        for i_2 in range(half_N):
            i_j = nei_ii[i_2]
            x_j = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_j, o_f, o_j)
            i_k = misc_cy.search_opposite_j_particle(coord_1d, direct_1d, nei_ii, x_i, i_j, x_j, box_length, o_f, o_k)

            x_k = misc_cy.calc_head_coordinate( coord_1d, direct_1d, i_k, o_f, o_k)
            x_i_j = misc_cy.calc_delta(x_j, x_i, box_length)
            x_i_k = misc_cy.calc_delta(x_k, x_i, box_length)
            x_ij_ik = [x_i_j[i] + x_i_k[i] for i in range(3)]

            sum_dist += np.dot(x_ij_ik, x_ij_ik)

        op_temp[name] = sum_dist / half_N

    return op_temp


def cpa_order_parameter_org(coord, direct, box_length, setting, neighbor_list, thread_num):
    a_times = setting['ave_times']
    o_factor = setting['o_factor']
    oi_oj_ok = setting['oi_oj_ok']

    # prepare parallel
    global coord_1d
    coord_1d = Array('d', misc_cy.convert_3dim_to_1dim(coord), lock=False)

    global direct_1d
    direct_1d = Array('d', misc_cy.convert_3dim_to_1dim(direct), lock=False)

    now_pool = Pool(thread_num)
    args = [[box_length, neighbor_list[i_i],
             i_i, o_factor, oi_oj_ok]
            for i_i in range(len(coord))]
    op_val_temp = now_pool.map(calc_c_wrapper, args)
    now_pool.close()

    del coord_1d
    del direct_1d

    op_value = misc_cy.data_num_name_to_data_name_num(op_val_temp, len(coord))

    comb = [(o_f, o_i, o_j, o_k)
            for o_f in o_factor for o_i in oi_oj_ok for o_j in oi_oj_ok for o_k in oi_oj_ok]

    # neighbor value averaging
    for a_t in range(a_times):
        for o_f, o_i, o_j, o_k in comb:
            name = misc_cy.naming('c', [a_t+1, o_f, o_i, o_j, o_k])
            name_old = misc_cy.naming('c', [a_t, o_f, o_i, o_j, o_k])
            op_value[name] = misc_cy.v_neighb_ave(
                neighbor_list, op_value[name_old])

    return op_value
