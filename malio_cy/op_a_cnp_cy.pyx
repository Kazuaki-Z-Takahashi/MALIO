# -*- coding: utf-8 -*-

from multiprocessing import Pool, Array
import numpy as np
import misc_cy
cimport cython
from malio_cy_def cimport *
from libcpp.string cimport string
from libcpp.pair cimport pair
cdef extern from "math.h" nogil:
    double sqrt(double x)

@cython.boundscheck(False)
@cython.cdivision(True)
def cnp_order_parameter(coord, direct, box_length, setting, neighbor_list, int thread_num):
    a_times = setting['ave_times']
    m_nei = setting['m_in_A']
    o_factor = setting['o_factor']
    oi_oj_ok = setting['oi_oj_ok']
    op_types = setting['op_types']
    comb = [[oo_f, oo_i, oo_j, oo_k]
        for oo_f in o_factor for oo_i in oi_oj_ok for oo_j in oi_oj_ok for oo_k in oi_oj_ok]

    cdef:
        int i_i, i_j, i_k, len_c = len(coord), o_f, o_i, o_j, o_k, i, j
        int l_neighbor_list_i_i, ii_j, ii_k, now_j, now_k, now_m, isop_type
        int opA, opN, opP, iic, max_m_nei = max(m_nei), idx, lop_types = len(op_types)
        double sum_r, x_ikn, x_jkn, dist, now_op
        string sop_type
        vector[double] dbox_length = box_length
        vector[vector[double]] dcoord = coord
        vector[vector[double]] ddirect = direct
        vector[vector[int]] ineighbor_list = neighbor_list
        vector[vector[int]] icomm = comb
        vector[int] im_nei = m_nei
        vector[string] vop_types
        dvec x_i = dvec()
        dvec x_j = dvec()
        dvec x_k = dvec()
        dvec r_ik = dvec()
        dvec r_jk = dvec()
        dvec r_ij = dvec()
        dvec r_kj = dvec()
        dvec sum_vec = dvec()
        dvec sum_vec_m = dvec()
        ivec i_j_nei = ivec()
        dvec i_j_dist = dvec()
        ivec_vec nei_ij = ivec_vec()

    for i in range(lop_types):
        vop_types.push_back(op_types[i].encode())
    op_val_temp = []
    for i_i in range(len_c):
        l_neighbor_list_i_i = ineighbor_list[i_i].size()
        x_i = dvec(3)
        x_j = dvec(3)
        x_k = dvec(3)
        r_ik = dvec(3)
        r_jk = dvec(3)
        r_ij = dvec(3)
        r_kj = dvec(3)
        sum_vec = dvec(3)
        sum_vec_m = dvec(3)

        op_temp = {}
        #### for o_f, o_i, o_j, o_k in comb:
        for iic in range(icomm.size()):
            o_f = icomm[iic][0]
            o_i = icomm[iic][1]
            o_j = icomm[iic][2]
            o_k = icomm[iic][3]
            ### x_i = misc.calc_head_coordinate(coord_1d, direct_1d, i_i, o_f, o_i)
            ### x_i = []
            for i in range(3):
                x_i[i] = dcoord[i_i][i] + o_f * o_i * ddirect[i_i][i]
                ### x_i.append(dcoord[i_i][i] + o_f * o_i * ddirect[i_i][i])

            ### args = [box_length, neighbor_list[i_i], x_i, o_f, o_j, o_k, max(m_nei)]
            ### nei_ij = misc.gen_neighbor_ijk(coord_1d, direct_1d, args)
            nei_ij = ivec_vec() ### []
            ### for i_j in neighbor_list[i_i]:
            for ii_j in range(l_neighbor_list_i_i):
                i_j = ineighbor_list[i_i][ii_j]
                ### args = [box_length, neighbor_list[i_i], x_i, i_j, o_f, o_j, o_k, max(m_nei)]
                ### i_j_nei = misc.gen_neighbor_ij(coord_1d, direct_1d, args)
                ### x_j = calc_head_coordinate(coord_1d, direct_1d, i_j, o_f, o_j)
                ### x_j = []
                for i in range(3):
                    x_j[i] = dcoord[i_j][i] + o_f * o_j * ddirect[i_j][i]
                    ### x_j.append(dcoord[i_j][i] + o_f * o_j * ddirect[i_j][i])
                i_j_nei = ivec() ### []
                i_j_dist = dvec() ### []
                ### for i_k in neighbor_list[i_i]:
                for ii_k in range(l_neighbor_list_i_i):
                    i_k = ineighbor_list[i_i][ii_k]
                    if i_j == i_k:
                        continue
                    ### x_k = calc_head_coordinate(coord_1d, direct_1d, i_k, o_f, o_k)
                    ### x_k = []
                    for i in range(3):
                        x_k[i] = dcoord[i_k][i] + o_f * o_k * ddirect[i_k][i]
                        ### x_k.append(dcoord[i_k][i] + o_f * o_k * ddirect[i_k][i])
                    ### dist = misc_cy.distance_ik_jk(x_i, x_j, dbox_length, x_k)
                    for i in range(3):
                        r_ik[i] = x_k[i] - x_i[i]
                        r_jk[i] = x_k[i] - x_j[i]
                        if r_ik[i] < -dbox_length[i] * 0.5:
                            r_ik[i] += dbox_length[i]
                        elif r_ik[i] >= dbox_length[i] * 0.5:
                            r_ik[i] -= dbox_length[i]
                        if r_jk[i] < -dbox_length[i] * 0.5:
                            r_jk[i] += dbox_length[i]
                        elif r_jk[i] >= dbox_length[i] * 0.5:
                            r_jk[i] -= dbox_length[i]
                    x_ikn = 0.0
                    x_jkn = 0.0
                    for i in range(3):
                        ### x_ikn += r_ik[i]*r_ik[i]
                        ### x_jkn += r_jk[i]*r_jk[i]
                        x_ikn = x_ikn + r_ik[i]*r_ik[i]
                        x_jkn = x_jkn + r_jk[i]*r_jk[i]
                    dist = sqrt(x_ikn) + sqrt(x_jkn)

                    ### [i_j_nei, i_j_dist] = misc.add_index_to_list(i_j_nei, i_j_dist, max(m_nei), dist, i_k)
                    ### def add_index_to_list(i_j_nei, dist_ij, size, dist, index):
                    ### if len(i_j_nei) < max(m_nei):
                    if i_j_nei.size() < max_m_nei:
                        ### i_j_nei.append(i_k)
                        ### i_j_dist.append(dist)
                        i_j_nei.push_back(i_k)
                        i_j_dist.push_back(dist)
                    else:
                        ### if max(i_j_dist) > dist:
                        ###     idx = i_j_dist.index(max(i_j_dist))
                        if misc_cy.vdmax(i_j_dist) > dist:
                            idx = misc_cy.ind_vdmax(i_j_dist)
                            i_j_nei[idx] = i_k
                            i_j_dist[idx] = dist

                ### [i_j_nei, i_j_dist] = misc_cy.sort_by_distance([i_j_nei], [i_j_dist])
                ### i_j_nei = i_j_nei[0]
                ### i_j_dist = i_j_dist[0]
                i_j_nei, i_j_dist = misc_cy.sort_by_distance_cy(i_j_nei, i_j_dist)
                ### nei_ij.append(i_j_nei)
                nei_ij.push_back(i_j_nei)

            ### for now_m in m_nei:
            for now_m in im_nei:
                ### for op_type in op_types:
                for sop_type in vop_types:
                    ### sum_vec_m = [0, 0, 0]
                    for i in range(3):
                        sum_vec_m[i] = 0.0
                    sum_r = 0.0
                    ### for now_j, i_j in enumerate(neighbor_list[i_i]):
                    opA = <int>(sop_type == b'A')
                    opN = <int>(sop_type == b'N')
                    opP = <int>(sop_type == b'P')
                    for now_j in range(l_neighbor_list_i_i):
                        i_j = ineighbor_list[i_i][now_j]
                        ### x_j = misc.calc_head_coordinate(coord_1d, direct_1d, i_j, o_f, o_j)
                        ### x_j = []
                        for i in range(3):
                            x_j[i] = dcoord[i_j][i] + o_f * o_j * ddirect[i_j][i]
                            ### x_j.append(dcoord[i_j][i] + o_f * o_j * ddirect[i_j][i])

                        ### sum_vec = [0.0, 0.0, 0.0]
                        for i in range(3):
                            sum_vec[i] = 0.0
                        for now_k in range(now_m):
                            ### if now_k >= len(nei_ij[now_j]):
                            if now_k >= nei_ij[now_j].size():
                                continue
                            i_k = nei_ij[now_j][now_k]
                            ### x_k = misc.calc_head_coordinate(coord_1d, direct_1d, i_k, o_f, o_k)
                            ### x_k = []
                            for i in range(3):
                                x_k[i] = dcoord[i_k][i] + o_f * o_k * ddirect[i_k][i]
                                ### x_k.append(dcoord[i_k][i] + o_f * o_k * ddirect[i_k][i])
                            if opA > 0: ### op_type is 'A':
                                ### r_ik = misc_cy.calc_delta(x_k, x_i, box_length)
                                ### r_jk = misc_cy.calc_delta(x_k, x_j, box_length)
                                for i in range(3):
                                    r_ik[i] = x_k[i] - x_i[i]
                                    r_jk[i] = x_k[i] - x_j[i]
                                    if r_ik[i] < -dbox_length[i] * 0.5:
                                        r_ik[i] += dbox_length[i]
                                    elif r_ik[i] >= dbox_length[i] * 0.5:
                                        r_ik[i] -= dbox_length[i]
                                    if r_jk[i] < -dbox_length[i] * 0.5:
                                        r_jk[i] += dbox_length[i]
                                    elif r_jk[i] >= dbox_length[i] * 0.5:
                                        r_jk[i] -= dbox_length[i]
                                ### sum_vec = [sum_vec[i] + r_ik[i] + r_jk[i] for i in range(3)]
                                for i in range(3):
                                    sum_vec[i] += r_ik[i] + r_jk[i]
                            elif opP > 0 or opN > 0: #### op_type is 'P' or op_type is 'N':
                                ### r_ij = misc_cy.calc_delta(x_j, x_i, box_length)
                                ### r_kj = misc_cy.calc_delta(x_j, x_k, box_length)
                                for i in range(3):
                                    r_ij[i] = x_j[i] - x_i[i]
                                    r_kj[i] = x_j[i] - x_k[i]
                                    if r_ij[i] < -dbox_length[i] * 0.5:
                                        r_ij[i] += dbox_length[i]
                                    elif r_ij[i] >= dbox_length[i] * 0.5:
                                        r_ij[i] -= dbox_length[i]
                                    if r_kj[i] < -dbox_length[i] * 0.5:
                                        r_kj[i] += dbox_length[i]
                                    elif r_kj[i] >= dbox_length[i] * 0.5:
                                        r_kj[i] -= dbox_length[i]
                                ### sum_vec = [sum_vec[i] + r_ij[i] + r_kj[i] for i in range(3)]
                                for i in range(3):
                                    sum_vec[i] += r_ij[i] + r_kj[i]
                        if opA > 0 or opP > 0: ### op_type is 'A' or op_type is 'P':
                            ### sum_r += np.dot(sum_vec, sum_vec)
                            for i in range(3):
                                sum_r += sum_vec[i]*sum_vec[i]
                        elif opN > 0: ### op_type is 'N':
                            ### sum_vec_m = [sum_vec_m[i] + sum_vec[i] for i in range(3)]
                            for i in range(3):
                                sum_vec_m[i] += sum_vec[i]

                    if opN > 0: ### op_type is 'N':
                        #### sum_r = np.dot(sum_vec_m, sum_vec_m)
                        for i in range(3):
                            ### sum_r += sum_vec_m[i]*sum_vec_m[i]
                            sum_r = sum_r + sum_vec_m[i]*sum_vec_m[i]
                    ### now_op = sum_r / float(len(neighbor_list[i_i]))
                    now_op = sum_r / <double>l_neighbor_list_i_i
                    name = misc_cy.naming('a', [0, sop_type.decode(), now_m, o_f, o_i, o_j, o_k])
                    op_temp[name] = now_op

        op_val_temp.append(op_temp)

    op_value = misc_cy.data_num_name_to_data_name_num(op_val_temp, len_c)

    comb = [(op_type, m, o_f, o_i, o_j, o_k)
            for op_type in op_types for m in m_nei
            for o_f in o_factor for o_i in oi_oj_ok for o_j in oi_oj_ok for o_k in oi_oj_ok]
    for a_t in range(a_times):
        for op_type, m_nei, o_f, o_i, o_j, o_k in comb:
            name = misc_cy.naming('a', [a_t + 1, op_type, m_nei, o_f, o_i, o_j, o_k])
            name_old = misc_cy.naming('a', [a_t, op_type, m_nei, o_f, o_i, o_j, o_k])
            op_value[name] = misc_cy.v_neighb_ave(neighbor_list, op_value[name_old])

    return op_value


def calc_cnp_wrapper(args):
    [box_length, neighbor_list_ii,
        i_i, m_nei, o_factor, oi_oj_ok, op_types] = args

    comb = [(o_f, o_i, o_j, o_k)
            for o_f in o_factor for o_i in oi_oj_ok for o_j in oi_oj_ok for o_k in oi_oj_ok]

    op_temp = {}
    for o_f, o_i, o_j, o_k in comb:
        x_i = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_i, o_f, o_i)
        args = [box_length, neighbor_list_ii, x_i, o_f, o_j, o_k, max(m_nei)]
        nei_ij = misc_cy.gen_neighbor_ijk(coord_1d, direct_1d, args)

        for now_m in m_nei:
            for op_type in op_types:
                sum_vec_m = [0, 0, 0]
                sum_r = 0.0
                for now_j, i_j in enumerate(neighbor_list_ii):
                    x_j = misc_cy.calc_head_coordinate(
                        coord_1d, direct_1d, i_j, o_f, o_j)

                    sum_vec = [0.0, 0.0, 0.0]
                    for now_k in range(now_m):
                        if now_k >= len(nei_ij[now_j]):
                            continue
                        i_k = nei_ij[now_j][now_k]
                        x_k = misc_cy.calc_head_coordinate(
                            coord_1d, direct_1d, i_k, o_f, o_k)
                        if op_type is 'A':
                            r_ik = misc_cy.calc_delta(x_k, x_i, box_length)
                            r_jk = misc_cy.calc_delta(x_k, x_j, box_length)
                            sum_vec = [sum_vec[i] + r_ik[i] + r_jk[i] for i in range(3)]
                        elif op_type is 'P' or op_type is 'N':
                            r_ij = misc_cy.calc_delta(x_j, x_i, box_length)
                            r_kj = misc_cy.calc_delta(x_j, x_k, box_length)
                            sum_vec = [sum_vec[i] + r_ij[i] + r_kj[i] for i in range(3)]
                    
                    if op_type is 'A' or op_type is 'P':
                        sum_r += np.dot(sum_vec, sum_vec)
                    elif op_type is 'N':
                        sum_vec_m = [sum_vec_m[i] + sum_vec[i] for i in range(3)]

                if op_type is 'N':
                    sum_r = np.dot(sum_vec_m, sum_vec_m)

                now_op = sum_r / float(len(neighbor_list_ii))
                name = misc_cy.naming('a', [0, op_type, now_m, o_f, o_i, o_j, o_k])
                op_temp[name] = now_op

    return op_temp


def cnp_order_parameter_org(coord, direct, box_length, setting, neighbor_list, thread_num):
    a_times = setting['ave_times']
    m_nei = setting['m_in_A']
    o_factor = setting['o_factor']
    oi_oj_ok = setting['oi_oj_ok']
    op_types = setting['op_types']

    # prepare parallel
    global coord_1d
    coord_1d = Array('d', misc_cy.convert_3dim_to_1dim(coord), lock=False)

    global direct_1d
    direct_1d = Array('d', misc_cy.convert_3dim_to_1dim(direct), lock=False)

    now_pool = Pool(thread_num)
    args = [[box_length, neighbor_list[i_i],
             i_i, m_nei, o_factor, oi_oj_ok, op_types] for i_i in range(len(coord))]
    op_temp = now_pool.map(calc_cnp_wrapper, args)
    now_pool.close()

    del coord_1d
    del direct_1d

    op_value = misc_cy.data_num_name_to_data_name_num(op_temp, len(coord))

    comb = [(op_type, m, o_f, o_i, o_j, o_k)
            for op_type in op_types for m in m_nei
            for o_f in o_factor for o_i in oi_oj_ok for o_j in oi_oj_ok for o_k in oi_oj_ok]
    for a_t in range(a_times):
        for op_type, m_nei, o_f, o_i, o_j, o_k in comb:
            name = misc_cy.naming('a', [a_t + 1, op_type, m_nei, o_f, o_i, o_j, o_k])
            name_old = misc_cy.naming('a', [a_t, op_type, m_nei, o_f, o_i, o_j, o_k])
            op_value[name] = misc_cy.v_neighb_ave(neighbor_list, op_value[name_old])

    return op_value
