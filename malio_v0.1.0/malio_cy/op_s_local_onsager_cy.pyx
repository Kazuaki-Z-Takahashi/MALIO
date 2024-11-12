# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
from multiprocessing import Pool, Array
import misc_cy
from scipy.special import legendre
from malio_cy_def cimport *
cimport cython

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef calc_order_param(direct, int n_leg, ref_vec=None):
    # second Legendre polynomial or Onsarger's order parameter
    # direct = [ [0,1,1], [1,2,3], ... ]

    # legendre function
    ### legend_fac = list(legendre(n_leg))
    legend_fac = misc_cy.legendre_coeff(n_leg)

    cdef:
        int i, j, len_d = len(direct), len_leg = len(legend_fac)
        dvec order_param = dvec()
        double temp, norm, invnorm, cos_theta
        double i_dir[3], dref_vec[3]
        vector[vector[double]] ddirect = direct
        vector[double] dlegend_fac = legend_fac

    if ref_vec[0] == None:
        for i in range(3):
            dref_vec[i] = 0.0
        for i in range(len_d):
            for j in range(3):
                dref_vec[j] += ddirect[i][j]
    else:
        for i in range(3):
            dref_vec[i] = ref_vec[i]
    
    norm = 0.0
    for i in range(3):
        norm += dref_vec[i]*dref_vec[i]
    invnorm = math.sqrt(1.0/norm)
    for i in range(3):
        dref_vec[i] = dref_vec[i]*invnorm

    for i in range(len_d):
        norm = 0.0
        for j in range(3):
            i_dir[j] = ddirect[i][j]
            norm += i_dir[j]*i_dir[j]
        invnorm = math.sqrt(1.0/norm)
        cos_theta = 0.0
        for j in range(3):
            cos_theta += dref_vec[j]*i_dir[j]*invnorm

        temp = 0.0
        for i in range(len_leg):
            # n = 2 : legend_fac = [1.5, 0.0, -0.5]
            # temp += dlegend_fac[i]*cos_theta**(n_leg-i)
            temp += dlegend_fac[i]*math.pow(cos_theta,n_leg-i)
        order_param.push_back(temp)

    return [order_param, dref_vec]

@cython.boundscheck(False)
@cython.cdivision(True)
def onsager_order_parameter(direct, setting, neighbor_list, thread_num):
    a_times = setting['ave_times']
    n_legendre = setting['n_in_S']

    cdef:
        int i, i_i, i_j, n_leg, l_direct = len(direct), l_order_param
        vector[vector[double]] ddirect = direct
        vector[vector[int]] ineighbor_list = neighbor_list
        double tmp = 0.0

    op_val_temp= []
    for i_i in range(l_direct):
        l_neighbor_list_i_i = len(ineighbor_list[i_i])

        # order parameter
        part_direct = []
        for i_j in neighbor_list[i_i]:
            part_direct.append(direct[i_j])

        op_temp = {}
        for n_leg in n_legendre:
            [order_param, rev_vec] = calc_order_param(part_direct, n_leg, ddirect[i_i])
            name = misc_cy.naming('s', [0, n_leg])
            tmp = 0.0
            l_order_param = len(order_param)
            for i in range(len(order_param)):
                tmp = tmp + order_param[i]
            op_temp[name] = tmp/<double>l_order_param
            ### op_temp[name] = np.average(order_param)

        op_val_temp.append(op_temp)

    op_data = misc_cy.data_num_name_to_data_name_num(op_val_temp, l_direct)

    for a_t in range(a_times):
        for n_leg in n_legendre:
            name = misc_cy.naming('s', [a_t+1, n_leg])
            name_old = misc_cy.naming('s', [a_t, n_leg])
            op_data[name] = misc_cy.v_neighb_ave(ineighbor_list, op_data[name_old])

    return op_data


def calc_order_param_org(direct, int n_leg, ref_vec=None):
    # second Legendre polynomial or Onsarger's order parameter
    # direct = [ [0,1,1], [1,2,3], ... ]

    # legendre function
    legend_fac = list(legendre(n_leg))

    if ref_vec == None:
        ref_vec = [0, 0, 0]
        for idirect in direct:
            temp = np.array(idirect)
            if np.dot(ref_vec, temp) < 0.0:
                ref_vec -= temp
            else:
                ref_vec += temp
        ref_vec = ref_vec / np.sqrt(np.dot(ref_vec, ref_vec))

    order_param = []
    for idirect in direct:
        # length = np.sqrt(np.dot(x_coord,x_coord))
        i_dir = np.array(idirect)
        i_dir = i_dir / np.linalg.norm(i_dir)
        cos_theta = np.dot(ref_vec, i_dir)

        temp = 0.0
        for i in range(len(legend_fac)):
            # n = 2 : legend_fac = [1.5, 0.0, -0.5]
            temp += legend_fac[i]*cos_theta**(n_leg-i)

        order_param.append(temp)

    return [order_param, ref_vec]


def calc_s_wrapper(args):
    [neighbor_list_ii, i_i, n_legendre] = args

    direct_ii = [direct_1d[3 * i_i + i] for i in range(3)]
    # order parameter
    part_direct = []
    for i_j in neighbor_list_ii:
        direct_i_j = [direct_1d[3 * i_j + i] for i in range(3)]
        part_direct.append(direct_i_j)

    op_temp = {}
    for n_leg in n_legendre:
        [order_param, rev_vec] = calc_order_param(
            part_direct, n_leg, direct_ii)
        name = misc_cy.naming('s', [0, n_leg])
        op_temp[name] = np.average(order_param)

    return op_temp


def onsager_order_parameter_org(direct, setting, neighbor_list, thread_num):
    a_times = setting['ave_times']
    n_legendre = setting['n_in_S']

    global direct_1d
    direct_1d = Array('d', misc_cy.convert_3dim_to_1dim(direct), lock=False)

    now_pool = Pool(thread_num)
    args = [[neighbor_list[i_i], i_i, n_legendre]
            for i_i in range(len(direct))]
    op_val_temp = now_pool.map(calc_s_wrapper, args)
    now_pool.close()

    del direct_1d
    op_data = misc_cy.data_num_name_to_data_name_num(op_val_temp, len(direct))

    for a_t in range(a_times):
        for n_leg in n_legendre:
            name = misc_cy.naming('s', [a_t+1, n_leg])
            name_old = misc_cy.naming('s', [a_t, n_leg])
            op_data[name] = misc_cy.v_neighb_ave(neighbor_list, op_data[name_old])

    return op_data
