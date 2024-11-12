# -*- coding: utf-8 -*-

### import math
from multiprocessing import Pool, Array
import numpy as np
import misc_cy
from scipy.special import legendre
cimport cython
from malio_cy_def cimport *

@cython.boundscheck(False)
@cython.cdivision(True)
def mcmillan_order_parameter(coord, direct, box_length, setting, neighbor_list, thread_num):
    a_times = setting['ave_times']
    o_factor = setting['o_factor']
    oi_oj = setting['oi_oj']
    layer_list = setting['d_in_T']
    n_legendre = setting['n_in_T']

    # [box_length, neighbor_list[i_i], i_i, o_factor, oi_oj, layer_list, n_legendre]
    # [box_length, neighbor_list_ii, i_i, o_factor, oi_oj, layer_list, n_legendre]

    cdef:
        int i_i, i_j, i_k, len_c = len(coord), o_f, o_i, o_j, i, j
        int l_neighbor_list_ii, n_leg, len_layer = len(layer_list)
        int len_legendre = len(n_legendre), len_leg, ii_j
        double s_part, pdot, nrm1, nrm2, dist_from_plane, cos_part, dist_layers
        double x_i[3]
        double x_j[3]
        double dplane_var[4]
        double *sum_r = <double*>malloc(len_layer*sizeof(double))
        vector[double] dbox_length = box_length
        vector[vector[double]] dcoord = coord
        vector[vector[double]] ddirect = direct
        vector[vector[int]] ineighbor_list = neighbor_list
        vector[double] dlayer_list = layer_list

    op_val_temp= []
    for i_i in range(len_c):
        l_neighbor_list_ii = len(neighbor_list[i_i])

        comb = [(o_f, o_i, o_j, n_leg)
                for o_f in o_factor for o_i in oi_oj for o_j in oi_oj for n_leg in n_legendre]

        op_temp = {}
        for o_f, o_i, o_j, n_leg in comb:
            ### direct_ii = [direct_1d[3 * i_i + i] for i in range(3)]
            ### x_i = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_i, o_f, o_i)
            for i in range(3):
                x_i[i] = dcoord[i_i][i] + <double>o_f * <double>o_i * ddirect[i_i][i]

            ### if np.linalg.norm(direct_ii) == 0.0:
            nrm1 = 0.0
            for i in range(3):
                nrm1 += ddirect[i_i][i]*ddirect[i_i][i]
            if math.sqrt(nrm1) <= 1.0e-12:
                ### plane_var = misc_cy.gen_z_plane(x_i, [0, 0, 1])
                dplane_var[0] = 0.0
                dplane_var[1] = 0.0
                dplane_var[2] = 1.0
                dplane_var[3] = -dplane_var[2]*x_i[2]
            else:
                ### plane_var = misc_cy.gen_z_plane(x_i, direct_ii)
                dplane_var[0] = ddirect[i_i][0]
                dplane_var[1] = ddirect[i_i][1]
                dplane_var[2] = ddirect[i_i][2]
                dplane_var[3] = -dplane_var[0]*x_i[0]-dplane_var[1]*x_i[1]-dplane_var[2]*x_i[2]

            ### sum_r = [0 for i in range(len(layer_list))]
            for i in range(len_layer):
                sum_r[i] = 0.0
            ### for i_j in neighbor_list[i_i]:
            for ii_j in range(l_neighbor_list_ii):
                i_j = ineighbor_list[i_i][ii_j]
                ### direct_i_j = [direct_1d[3 * i_j + i] for i in range(3)]
                ### x_j = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_j, o_f, o_j)
                for i in range(3):
                    x_j[i] = dcoord[i_j][i] + <double>o_f * <double>o_j * ddirect[i_j][i]

                ### cos_theta = np.dot(direct_ii, direct_i_j)/(np.linalg.norm(direct_ii)*
                ###             np.linalg.norm(direct_i_j))
                pdot = 0.0
                nrm1 = 0.0
                nrm2 = 0.0
                for i in range(3):
                    pdot += ddirect[i_i][i]*ddirect[i_j][i]
                    nrm1 += ddirect[i_i][i]*ddirect[i_i][i]
                    nrm2 += ddirect[i_j][i]*ddirect[i_j][i]
                    cos_theta = pdot/math.sqrt(nrm1*nrm2)
                # legendre function
                ### legend_fac = list(legendre(n_leg))
                legend_fac = misc_cy.legendre_coeff(n_leg)
                len_leg = len(legend_fac)
            
                ### s_part = 0
                s_part = 0.0
                ### for i in range(len(legend_fac)):
                for i in range(len_leg):
                    ### s_part += legend_fac[i]*cos_theta**(n_leg-i)
                    s_part += legend_fac[i]*math.pow(cos_theta,n_leg-i)

                ### dist_from_plane = misc_cy.plane_point_distance(plane_var, box_length, x_j)
                dist_from_plane = misc_cy.dplane_point_distance(dplane_var, dbox_length, x_j)

                ### for i_k, dist_layers in enumerate(layer_list):
                for i_k in range(len_layer):
                    dist_layers = dlayer_list[i_k]
                    cos_part = math.cos( 2.0 * math.pi * dist_from_plane / dist_layers)
                    sum_r[i_k] += cos_part * s_part

            ### for i_k, dist_layers in enumerate(layer_list):
            ###     if not neighbor_list[i_i]:
            ###         sum_r[i_k] = 0.0
            ###     else:
            ###         sum_r[i_k] = sum_r[i_k] / (float(len(neighbor_list[i_i])))
            for i_k in range(len_layer):
                dist_layers = dlayer_list[i_k]
                if l_neighbor_list_ii == 0:
                    sum_r[i_k] = 0.0
                else:
                    sum_r[i_k] = sum_r[i_k] / <double>l_neighbor_list_ii
                name = misc_cy.naming('t', [0, o_f, o_i, o_j, dist_layers, n_leg])
                op_temp[name] = sum_r[i_k]

        op_val_temp.append(op_temp)

    op_data = misc_cy.data_num_name_to_data_name_num(op_val_temp, len_c)

    comb = [(o_f, o_i, o_j, dist_layers, n_leg)
            for o_f in o_factor for o_i in oi_oj
            for o_j in oi_oj for dist_layers in layer_list 
            for n_leg in n_legendre]
    for a_t in range(a_times):
        for o_f, o_i, o_j, dist_layers, n_leg in comb:
            name = misc_cy.naming('t', [a_t+1, o_f, o_i, o_j, dist_layers, n_leg])
            name_old = misc_cy.naming('t', [a_t, o_f, o_i, o_j, dist_layers, n_leg])
            op_data[name] = misc_cy.v_neighb_ave(neighbor_list, op_data[name_old])

    return op_data

def calc_t_wrapper(args):
    [box_length, neighbor_list_ii, i_i, o_factor, oi_oj, layer_list, n_legendre] = args
    
    comb = [(o_f, o_i, o_j, n_leg)
            for o_f in o_factor for o_i in oi_oj for o_j in oi_oj for n_leg in n_legendre]

    op_temp = {}
    for o_f, o_i, o_j, n_leg in comb:
        direct_ii = [direct_1d[3 * i_i + i] for i in range(3)]
        x_i = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_i, o_f, o_i)

        if np.linalg.norm(direct_ii) == 0.0:
            plane_var = misc_cy.gen_z_plane(x_i, [0, 0, 1])
        else:
            plane_var = misc_cy.gen_z_plane(x_i, direct_ii)

        sum_r = [0 for i in range(len(layer_list))]
        for i_j in neighbor_list_ii:
            direct_i_j = [direct_1d[3 * i_j + i] for i in range(3)]
            x_j = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_j, o_f, o_j)

            cos_theta = np.dot(direct_ii, direct_i_j)/(np.linalg.norm(direct_ii)*np.linalg.norm(direct_i_j))
            # legendre function
            legend_fac = list(legendre(n_leg))
            
            s_part = 0
            for i in range(len(legend_fac)):
                # n = 2 : legend_fac = [1.5, 0.0, -0.5]
                s_part += legend_fac[i]*cos_theta**(n_leg-i)

            dist_from_plane = misc_cy.plane_point_distance(
                plane_var, box_length, x_j)

            for i_k, dist_layers in enumerate(layer_list):
                cos_part = math.cos( 2.0 * math.pi * dist_from_plane / dist_layers)
                sum_r[i_k] += cos_part * s_part

        for i_k, dist_layers in enumerate(layer_list):
            if not neighbor_list_ii:
                sum_r[i_k] = 0.0
            else:
                sum_r[i_k] = sum_r[i_k] / (
                    float(len(neighbor_list_ii)))

            name = misc_cy.naming('t', [0, o_f, o_i, o_j, dist_layers, n_leg])
            op_temp[name] = sum_r[i_k]

    return op_temp


def mcmillan_order_parameter_org(coord, direct, box_length, setting, neighbor_list, thread_num):
    a_times = setting['ave_times']
    o_factor = setting['o_factor']
    oi_oj = setting['oi_oj']
    layer_list = setting['d_in_T']
    n_legendre = setting['n_in_T']

    # prepare parallel
    global coord_1d
    coord_1d = Array('d', misc_cy.convert_3dim_to_1dim(coord), lock=False)

    global direct_1d
    direct_1d = Array('d', misc_cy.convert_3dim_to_1dim(direct), lock=False)

    now_pool = Pool(thread_num)
    args = [[box_length, neighbor_list[i_i],
             i_i, o_factor, oi_oj, layer_list, n_legendre] for i_i in range(len(coord))]
    op_val_temp = now_pool.map(calc_t_wrapper, args)
    now_pool.close()

    del coord_1d
    del direct_1d

    op_data = misc_cy.data_num_name_to_data_name_num(op_val_temp, len(coord))

    comb = [(o_f, o_i, o_j, dist_layers, n_leg)
            for o_f in o_factor for o_i in oi_oj
            for o_j in oi_oj for dist_layers in layer_list 
            for n_leg in n_legendre]
    for a_t in range(a_times):
        for o_f, o_i, o_j, dist_layers, n_leg in comb:
            name = misc_cy.naming('t', [a_t+1, o_f, o_i, o_j, dist_layers, n_leg])
            name_old = misc_cy.naming('t', [a_t, o_f, o_i, o_j, dist_layers, n_leg])
            op_data[name] = misc_cy.v_neighb_ave(neighbor_list, op_data[name_old])

    return op_data
