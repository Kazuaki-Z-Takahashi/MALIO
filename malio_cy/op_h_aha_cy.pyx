# -*- coding: utf-8 -*-

### import math
from multiprocessing import Pool, Array
from scipy.fftpack import fft
import numpy as np
import misc_cy
cimport cython
from malio_cy_def cimport *
from libcpp.string cimport string

def histogram_normalize(hist):
    sum_hist = np.sum(hist)
    hist = [hist[i]/sum_hist for i in range(len(hist))]
    return hist

@cython.boundscheck(False)
@cython.cdivision(True)
cdef dvec histogram_normalize_cy(dvec hist):
    cdef:
        int i, l_hist = len(hist)
        double sum_hist = 0.0
    for i in range(l_hist):
        sum_hist += hist[i]
    for i in range(l_hist):
        hist[i] /= sum_hist
    return hist

@cython.boundscheck(False)
@cython.cdivision(True)
def aha_order_parameter(coord, direct, box_length, setting, neighbor_list, thread_num):
    a_times = setting['ave_times']
    b_times = setting['b_in_H']
    h_num = setting['hist_num']
    o_factor = setting['o_factor']
    oi_oj_ok = setting['oi_oj_ok']
    nu = setting['nu']

    cdef:
        int i_i, i_j, i_k, len_c = len(coord), o_f, o_i, o_j, o_k, i, j, k
        int l_neighbor_list_i_i, i_2, len_h_num = len(h_num), h_i, now_i
        int n_list, l_neighbor_list = len(neighbor_list), ii_j, inei, b_t, name_indx
        int l_h_hist_name_old_i_i, l_h_hist_name_i_i, l_nu = len(nu), inu, name_old_indx
        double theta, pdot, nrm1, nrm2, increment, avpart, inv_l_neighbor_list_i_i_1
        double mathpi = math.pi, zr, zi
        double x_i[3]
        double x_j[3]
        double x_k[3]
        double x_i_j[3]
        double x_ik[3]
        vector[double] dbox_length = box_length
        vector[vector[double]] dcoord = coord
        vector[vector[double]] ddirect = direct
        vector[vector[int]] ineighbor_list = neighbor_list
        int *lneighbor_list = <int*>malloc(len_c*sizeof(int))
        int *ih_num = <int*>malloc(len_h_num*sizeof(int))
        int *intnu = <int*>malloc(l_nu*sizeof(int))
        dvec val_ave = dvec(l_neighbor_list)
        dvec dv = dvec()
        dvec_vec vhist_temp = dvec_vec(len_h_num, dv)
        dvec_vec dg_list = dvec_vec(len_c, dv)
        dvec_vec dval = dvec_vec()
        dvec_vec dvv = dvec_vec(len_c, dv)
        dvec_vec_vec vh_hist = dvec_vec_vec()
    
    for i in range(len_c):
        lneighbor_list[i] = len(neighbor_list[i])
    for i in range(l_nu):
        intnu[i] = nu[i]
    for i in range(len_h_num):
        ih_num[i] = h_num[i]
    for i in range(len_h_num):
        for j in range(ih_num[i]):
            vhist_temp[i].push_back(0.0)
    for i_i in range(len_c):
        for i in range(l_nu):
            dg_list[i_i].push_back(0.0)

    op_val_temp= []
    for i_i in range(len_c):
        l_neighbor_list_i_i = lneighbor_list[i_i]

        comb = [(o_f, o_i, o_j, o_k)
            for o_f in o_factor for o_i in oi_oj_ok for o_j in oi_oj_ok for o_k in oi_oj_ok]

        op_temp = {}
        for o_f, o_i, o_j, o_k in comb:
            # init histogram
            ### hist_temp = []  # hist_temp[bin_num][ibin]
            ### for i_k, _ in enumerate(h_num):
            ###     h_temp2 = [0 for i in range(h_num[i_k])]
            ###     hist_temp.append(h_temp2)
            for i_k in range(len_h_num):
                for i in range(ih_num[i_k]):
                    vhist_temp[i_k][i] = 0.0

            ### x_i = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_i, o_f, o_i)
            for i in range(3):
                x_i[i] =dcoord[i_i][i] + <double>o_f * <double>o_i * ddirect[i_i][i]

            n_list = l_neighbor_list_i_i
            increment = 1.0 / <double>(n_list*(n_list-1)/2)
            for i_2 in range(l_neighbor_list_i_i - 1):
                i_j = ineighbor_list[i_i][i_2]
                ### x_j = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_j, o_f, o_j)
                for i in range(3):
                    x_j[i] = dcoord[i_j][i] + <double>o_f * <double>o_j * ddirect[i_j][i]
                for i_3 in range(i_2 + 1, l_neighbor_list_i_i):
                    i_k = ineighbor_list[i_i][i_3]
                    ### x_k = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_k, o_f, o_k)
                    for i in range(3):
                        x_k[i] = dcoord[i_k][i] + <double>o_f * <double>o_k * ddirect[i_k][i]

                    ### x_i_j = misc_cy.calc_delta(x_j, x_i, box_length)
                    ### x_ik = misc_cy.calc_delta(x_k, x_i, box_length)
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
                    ### if math.sqrt(nrm1*nrm2) > 1.0e-10:
                    if nrm1*nrm2 > 1.0e-12:
                        theta = math.acos(pdot/math.sqrt(nrm1*nrm2))
                    else:
                        theta = 0.0
                    ### try:
                    ###     theta = misc_cy.angle(x_i_j, x_ik)
                    ### except ValueError:
                    ###     theta = 0.0
                    if theta >= mathpi:
                        theta -= mathpi

                    ### for i_k, _ in enumerate(h_num):
                    ###     h_i = h_num[i_k]
                    ###     now_i = int(h_i * theta / math.pi)
                    ###     hist_temp[i_k][now_i] += 1.0 / float(n_list * (n_list - 1) / 2)
                    for i_k in range(len_h_num):
                        h_i = ih_num[i_k]
                        now_i = <int>(<double>h_i * theta / mathpi)
                        vhist_temp[i_k][now_i] += increment

            ### for i_k, _ in enumerate(h_num):
            for i_k in range(len_h_num):
                name = misc_cy.naming('h', [0, 0, h_num[i_k], o_f, o_i, o_j, o_k])
                ### op_temp[name] = hist_temp[i_k]
                op_temp[name] = vhist_temp[i_k]

        ### return op_temp
        op_val_temp.append(op_temp)

    h_hist = misc_cy.data_num_name_to_data_name_num(op_val_temp, len_c)

    ### fast access preparation
    vname = []
    for i, name in enumerate(h_hist):
        vh_hist.push_back(dvv)
        vname.append(name)
        for j in range(len_c):
            vh_hist[i].push_back(dv)
            for k in range(len(h_hist[name][j])):
                vh_hist[i][j].push_back(h_hist[name][j][k])

    # neighbor histogram averaging
    comb = [(o_f, o_i, o_j, o_k)
            for o_f in o_factor for o_i in oi_oj_ok for o_j in oi_oj_ok for o_k in oi_oj_ok]

    for o_f, o_i, o_j, o_k in comb:
        for b_t in range(b_times):
            ### for i_k, _ in enumerate(h_num):
            for i_k in range(len_h_num):
                name = misc_cy.naming('h', [0, b_t+1, h_num[i_k], o_f, o_i, o_j, o_k])
                name_old = misc_cy.naming('h', [0, b_t, h_num[i_k], o_f, o_i, o_j, o_k])
                vname.append(name)
                name_indx = vname.index(name)
                name_old_indx = vname.index(name_old)
                vh_hist.push_back(dvv)
                ### h_hist[name] = misc_cy.v_neighb_ave(neighbor_list, h_hist[name_old])
                ### for i_i, _ in enumerate(neighbor_list):
                for i_i in range(l_neighbor_list):
                    vh_hist[name_indx].push_back(dv)
                    l_neighbor_list_i_i = lneighbor_list[i_i]
                    l_h_hist_name_old_i_i = len(h_hist[name_old][i_i])
                    inv_l_neighbor_list_i_i_1 = 1.0 / <double>(l_neighbor_list_i_i + 1)
                    ### part = [0 for _ in range(len(h_hist[name_old][i_i]))]
                    for i_j in range(l_h_hist_name_old_i_i):
                        ### for inei in neighbor_list[i_i] + [i_i]:
                        pdot = 0.0
                        for ii_j in range(l_neighbor_list_i_i):
                            inei = ineighbor_list[i_i][ii_j]
                            pdot += vh_hist[name_old_indx][inei][i_j]
                        pdot += vh_hist[name_old_indx][i_i][i_j]
                        vh_hist[name_indx][i_i].push_back(pdot*inv_l_neighbor_list_i_i_1)

    # FFT
    h_data_part_nu = {}
    for o_f, o_i, o_j, o_k in comb:
        for b_t in range(b_times + 1):
            ### for i_k, _ in enumerate(h_num):
            for i_k in range(len_h_num):
                name = misc_cy.naming('h', [0, b_t, h_num[i_k], o_f, o_i, o_j, o_k])
                name_indx = vname.index(name)
                ### g_list = []
                ### for i_i, _ in enumerate(coord):
                for i_i in range(len_c):
                    ### l_h_hist_name_i_i = len(h_hist[name][i_i])
                    ### g_func = fft(histogram_normalize(h_hist[name][i_i]))
                    l_h_hist_name_i_i = len(vh_hist[name_indx][i_i])
                    g_func = fft(histogram_normalize_cy(vh_hist[name_indx][i_i]))
                    g_power = np.abs(g_func)**2
                    ### power = []
                    ### for inu in nu:
                    ###     try:
                    ###         power.append(g_power[inu])
                    ###     except IndexError as e:
                    ###        power.append(0)
                    ### g_list.append(power)
                    for i in range(l_nu):
                        inu = intnu[i]
                        if inu >= 0 and inu < l_h_hist_name_i_i:
                            dg_list[i_i][i] = g_power[inu]
                        else:
                            dg_list[i_i][i] = 0.0
                ### h_data_part_nu[name] = g_list
                h_data_part_nu[name] = dg_list

    op_value = {}
    for o_f, o_i, o_j, o_k in comb:
        for b_t in range(b_times + 1):
            ### for i_k, _ in enumerate(h_num):
            for i_k in range(len_h_num):
                for i_l, _ in enumerate(nu):
                    inu = nu[i_l]
                    name = misc_cy.naming('h', [0, b_t, h_num[i_k], o_f, o_i, o_j, o_k])
                    name_h = name + '_nu=' + str(inu)
                    op_value[name_h] = [h_data_part_nu[name][i_i][i_l] for i_i in range(len_c)]

    j = 0
    for b_t in range(b_times + 1):
        for i_k in range(len_h_num):
            for o_f, o_i, o_j, dist_layers in comb:
                for i_l, _ in enumerate(nu):
                    dval.push_back(dv)
                    inu = nu[i_l] ### ohino
                    name = misc_cy.naming('h', [0, b_t, h_num[i_k], o_f, o_i, o_j, o_k])
                    name_h = name + '_nu=' + str(inu)
                    for i_i in range(len_c):
                        dval[j].push_back(op_value[name_h][i_i])
                    j += 1           

    for a_t in range(a_times):
        j = 0
        for b_t in range(b_times + 1):
            ### for i_k, _ in enumerate(h_num):
            for i_k in range(len_h_num):
                for o_f, o_i, o_j, dist_layers in comb:
                    for i_l, _ in enumerate(nu):
                        ### inu = nu[i_l] ### ohino
                        name = misc_cy.naming('h', [a_t+1, b_t, h_num[i_k], o_f, o_i, o_j, dist_layers]) + '_nu=' + str(inu)
                        name_old = misc_cy.naming('h', [a_t, b_t, h_num[i_k], o_f, o_i, o_j, dist_layers]) + '_nu=' + str(inu)
                        ### op_value[name] = misc_cy.v_neighb_ave(neighbor_list, op_value[name_old])
                        for i_i in range(l_neighbor_list):
                            l_neighbor_list_i_i = lneighbor_list[i_i]
                            avpart = 0.0
                            for ii_j in range(l_neighbor_list_i_i):
                                inei = ineighbor_list[i_i][ii_j]
                                avpart += dval[j][inei] # dval[inei]
                            avpart += dval[j][i_i] # dval[i_i]
                            val_ave[i_i] = avpart/<double>(l_neighbor_list_i_i + 1)
                        j += 1
                        op_value[name] = val_ave

    free(lneighbor_list)
    free(intnu)
    free(ih_num)
    return op_value


def calc_h_wrapper(args):
    [box_length, neighbor_list_ii,
        i_i, h_num, o_factor, oi_oj_ok] = args

    comb = [(o_f, o_i, o_j, o_k)
            for o_f in o_factor for o_i in oi_oj_ok for o_j in oi_oj_ok for o_k in oi_oj_ok]

    op_temp = {}
    for o_f, o_i, o_j, o_k in comb:
        # init histogram
        hist_temp = []  # hist_temp[bin_num][ibin]
        for i_k, _ in enumerate(h_num):
            h_temp2 = [0 for i in range(h_num[i_k])]
            hist_temp.append(h_temp2)

        x_i = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_i, o_f, o_i)

        n_list = len(neighbor_list_ii)
        for i_2 in range(len(neighbor_list_ii) - 1):
            i_j = neighbor_list_ii[i_2]
            x_j = misc_cy.calc_head_coordinate(coord_1d, direct_1d, i_j, o_f, o_j)
            for i_3 in range(i_2 + 1, len(neighbor_list_ii)):
                i_k = neighbor_list_ii[i_3]
                x_k = misc_cy.calc_head_coordinate(
                    coord_1d, direct_1d, i_k, o_f, o_k)

                x_i_j = misc_cy.calc_delta(x_j, x_i, box_length)
                x_ik = misc_cy.calc_delta(x_k, x_i, box_length)
                try:
                    theta = misc_cy.angle(x_i_j, x_ik)
                except ValueError:
                    theta = 0.0
                if theta >= math.pi:
                    theta -= math.pi

                for i_k, _ in enumerate(h_num):
                    h_i = h_num[i_k]
                    now_i = int(h_i * theta / math.pi)
                    hist_temp[i_k][now_i] += 1.0 / float(
                        n_list * (n_list - 1) / 2)

        for i_k, _ in enumerate(h_num):
            name = misc_cy.naming('h', [0, 0, h_num[i_k], o_f, o_i, o_j, o_k])
            op_temp[name] = hist_temp[i_k]

    return op_temp


def aha_order_parameter_org(coord, direct, box_length, setting, neighbor_list, thread_num):
    a_times = setting['ave_times']
    b_times = setting['b_in_H']
    h_num = setting['hist_num']
    o_factor = setting['o_factor']
    oi_oj_ok = setting['oi_oj_ok']
    nu = setting['nu']

    # prepare parallel
    global coord_1d
    coord_1d = Array('d', misc_cy.convert_3dim_to_1dim(coord), lock=False)

    global direct_1d
    direct_1d = Array('d', misc_cy.convert_3dim_to_1dim(direct), lock=False)

    now_pool = Pool(thread_num)
    args = [[box_length, neighbor_list[i_i], i_i, h_num, o_factor, oi_oj_ok]
            for i_i in range(len(coord))]
    op_val_temp = now_pool.map(calc_h_wrapper, args)
    now_pool.close()

    del coord_1d
    del direct_1d

    h_hist = misc_cy.data_num_name_to_data_name_num(op_val_temp, len(coord))

    # neighbor histogram averaging
    comb = [(o_f, o_i, o_j, o_k)
            for o_f in o_factor for o_i in oi_oj_ok for o_j in oi_oj_ok for o_k in oi_oj_ok]
    for o_f, o_i, o_j, o_k in comb:
        for b_t in range(b_times):
            for i_k, _ in enumerate(h_num):
                name = misc_cy.naming('h', [0, b_t+1, h_num[i_k], o_f, o_i, o_j, o_k])
                name_old = misc_cy.naming(
                    'h', [0, b_t, h_num[i_k], o_f, o_i, o_j, o_k])
                h_hist[name] = misc_cy.v_neighb_ave(
                    neighbor_list, h_hist[name_old])

    # FFT
    h_data_part_nu = {}
    for o_f, o_i, o_j, o_k in comb:
        for b_t in range(b_times + 1):
            for i_k, _ in enumerate(h_num):
                name = misc_cy.naming('h', [0, b_t, h_num[i_k], o_f, o_i, o_j, o_k])
                g_list = []
                for i_i, _ in enumerate(coord):
                    g_func = fft(histogram_normalize(h_hist[name][i_i]))
                    g_power = np.abs(g_func)**2
                    power = []
                    for inu in nu:
                        try:
                            power.append(g_power[inu])
                        except IndexError as e:
                            power.append(0)
                    g_list.append(power)
                h_data_part_nu[name] = g_list

    op_value = {}
    for o_f, o_i, o_j, o_k in comb:
        for b_t in range(b_times + 1):
            for i_k, _ in enumerate(h_num):
                for i_l, _ in enumerate(nu):
                    inu = nu[i_l]
                    name = misc_cy.naming('h', [0, b_t, h_num[i_k], o_f, o_i, o_j, o_k])
                    name_h = name + '_nu=' + str(inu)

                    op_value[name_h] = [h_data_part_nu[name][i_i][i_l]
                                      for i_i in range(len(coord))]

    for a_t in range(a_times):
        for b_t in range(b_times + 1):
            for i_k, _ in enumerate(h_num):
                for o_f, o_i, o_j, dist_layers in comb:
                    for i_l, _ in enumerate(nu):
                        inu = nu[i_l] ### ohino
                        name = misc_cy.naming('h', [a_t+1, b_t, h_num[i_k], o_f, o_i, o_j, dist_layers]) + '_nu=' + str(inu)
                        name_old = misc_cy.naming('h', [a_t, b_t, h_num[i_k], o_f, o_i, o_j, dist_layers]) + '_nu=' + str(inu)
                        op_value[name] = misc_cy.v_neighb_ave(neighbor_list, op_value[name_old])

    return op_value
