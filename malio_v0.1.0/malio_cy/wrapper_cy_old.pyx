# -*- coding: utf-8 -*-
import time
import numpy as np

import malio_cy.misc_cy as misc
import malio_cy.neighbor_build_cy as neighbor_build
import malio_cy.op_s_local_onsager_cy as op_s_local_onsager
import malio_cy.op_a_cnp_cy as op_a_cnp
import malio_cy.op_b_baa_cy as op_b_baa
import malio_cy.op_c_cpa_cy as op_c_cpa
import malio_cy.op_d_nda_cy as op_d_nda
import malio_cy.op_f_afs_cy as op_f_afs
import malio_cy.op_i_top_cy as op_i_top
import malio_cy.op_t_msigma_cy as op_t_msigma
import malio_cy.op_h_aha_cy as op_h_aha
import malio_cy.op_q_spherical_cy as op_q_spherical
import malio_cy.op_w_wigner_cy as op_w_wigner
import malio_cy.op_z_user1_define_cy as op_z_user1_define

def param_check(op_settings, idx, init_value):
    if idx not in op_settings:
        op_settings[idx] = init_value
    return op_settings

def param_check_all(op_settings):
    
    op_settings = param_check(op_settings, 'ave_times', 0)
    op_settings = param_check(op_settings, 'oi_oj'    , [0])
    op_settings = param_check(op_settings, 'o_factor' , [0])
    # A
    if 'A' in op_settings['analysis_type']:
        op_settings = param_check(op_settings, 'm_in_A' , [2])
        op_settings = param_check(op_settings, 'types_in_A' , ['A'])
    # B
    if 'B' in op_settings['analysis_type']:
        op_settings = param_check(op_settings, 'm_in_B' , [2])
        op_settings = param_check(op_settings, 'n_in_B' , [1])
        op_settings = param_check(op_settings, 'phi_in_B' , [0])
    # C
    # D
    # H
    if 'H' in op_settings['analysis_type']:
        op_settings = param_check(op_settings, 'b_in_H' , 0)
        op_settings = param_check(op_settings, 'bin_in_H' , [12])
        op_settings = param_check(op_settings, 'nu_in_H' , [3])
    # I
    # Q
    if 'Q' in op_settings['analysis_type']:
        op_settings = param_check(op_settings, 'l_in_Q' , [4])
        op_settings = param_check(op_settings, 'b_in_Q' , 0)
        op_settings = param_check(op_settings, 'p_in_Q' , [0])
    # W
    if 'W' in op_settings['analysis_type']:
        op_settings = param_check(op_settings, 'l_in_Q' , [4])
        op_settings = param_check(op_settings, 'b_in_Q' , 0)
        op_settings = param_check(op_settings, 'p_in_Q' , [0])
    # S
    if 'S' in op_settings['analysis_type']:
        op_settings = param_check(op_settings, 'n_in_S' , [2])
    # T
    if 'T' in op_settings['analysis_type']:
        op_settings = param_check(op_settings, 'n_in_T' , [2])
    return op_settings

def op_analyze_with_neighbor_list(coord, direct, box_length, NR_name, op_settings, n_list, op_data, thread_num):
    """ Analyze structure
    :param method: method for analysis.
    Please see the details in readme file.
    :param coord: = [[0,0,0],[1,0,0]]
    :param direct: = [[1,0,0],[1,0,0]]
    :param box_length: = [10,10,10]
    :param NR_name: = 'N6'
    :param n_list: is neighbor list [[1],[0]]
    :param op_settings:
    :param thread_num:
    :return op_data: type(op_data) is dict.

    """

    op_settings = param_check_all(op_settings)
    op_temp = {}

    # common neighborhood parameter (CNP) A
    if 'A' in op_settings['analysis_type']:
        t_start = time.time()
        setting = {'ave_times': op_settings['ave_times'],
                       'm_in_A': op_settings['m_in_A'],
                       'o_factor': op_settings['o_factor'],
                       'oi_oj_ok': op_settings['oi_oj'],
                       'op_types': op_settings['types_in_A']}
        op_temp['A_' + NR_name] = op_a_cnp.cnp_order_parameter(
            coord, direct, box_length, setting, n_list, thread_num)
        t_end = time.time()
        print("# CNP A elap time ", t_end - t_start)

    # bond angle analysis (BAA) B
    if 'B' in op_settings['analysis_type']:
        t_start = time.time()
        setting = {'ave_times': op_settings['ave_times'],
                       'm': op_settings['m_in_B'],
                       'phi': op_settings['phi_in_B'],
                       'n': op_settings['n_in_B'],
                       'o_factor': op_settings['o_factor'],
                       'oi_oj_ok': op_settings['oi_oj']
                       }
        op_temp['B_' + NR_name] = op_b_baa.baa_order_parameter(
            coord, direct, box_length, setting, n_list, thread_num)
        t_end = time.time()
        print("# BAA B elap time ", t_end - t_start)

    # centrometry parameter analysis (CPA) C
    if 'C' in op_settings['analysis_type']:
        t_start = time.time()
        setting = {'ave_times': op_settings['ave_times'],
                       'o_factor': op_settings['o_factor'],
                       'oi_oj_ok': op_settings['oi_oj']}
        op_temp['C_' + NR_name] = op_c_cpa.cpa_order_parameter(
            coord, direct, box_length, setting, n_list, thread_num)
        t_end = time.time()
        print("# CPA C elap time ", t_end - t_start)

    # neighbor distance analysis (NDA) D
    if 'D' in op_settings['analysis_type']:
        t_start = time.time()
        setting = {'ave_times': op_settings['ave_times'],
                       'o_factor': op_settings['o_factor'],
                       'oi_oj_ok': op_settings['oi_oj'],
                       'func': op_settings['function']}
        op_temp['D_' + NR_name] = op_d_nda.nda_order_parameter(
            coord, direct, box_length, setting, n_list, thread_num)
        t_end = time.time()
        print("# NDA D elap time ", t_end - t_start)

    # Angular Fourier Series like parameter (AFS) F
    if 'F' in op_settings['analysis_type']:
        t_start = time.time()
        setting = {'ave_times': op_settings['ave_times'],
                       'o_factor': op_settings['o_factor'],
                       'oi_oj_ok': op_settings['oi_oj'],
                       'func': op_settings['function'],
                       'l_in_F': op_settings['l_in_F']}
        op_temp['F_' + NR_name] = op_f_afs.afs_order_parameter(
            coord, direct, box_length, setting, n_list, thread_num)
        t_end = time.time()
        print("# AFS F elap time ", t_end - t_start)

    # angle histogram analysis (AHA) H
    if 'H' in op_settings['analysis_type']:
        t_start = time.time()
        setting = {'ave_times': op_settings['ave_times'],
                       'o_factor': op_settings['o_factor'],
                       'oi_oj_ok': op_settings['oi_oj'],
                       'b_in_H': op_settings['b_in_H'],
                       'hist_num': op_settings['bin_in_H'],
                       'nu': op_settings['nu_in_H']
                       }
        op_temp['H_' + NR_name] = op_h_aha.aha_order_parameter(
            coord, direct, box_length, setting, n_list, thread_num)
        t_end = time.time()
        print("# AHA H elap time ", t_end - t_start)

    # tetrahedral order parameter (TOP) I
    if 'I' in op_settings['analysis_type']:
        t_start = time.time()
        setting = {'ave_times': op_settings['ave_times'],
                       'o_factor': op_settings['o_factor'],
                       'oi_oj_ok': op_settings['oi_oj']}
        op_temp['I_' + NR_name] = op_i_top.top_order_parameter(
            coord, direct, box_length, setting, n_list, thread_num)
        t_end = time.time()
        print("# TOP I elap time ", t_end - t_start)
    
    # Spherical Order parameter Q
    if 'Q' in op_settings['analysis_type']:
        t_start = time.time()
        setting = {'ave_times': op_settings['ave_times'],
                       'oi_oj': op_settings['oi_oj'],
                       'o_factor': op_settings['o_factor'],
                       'b_in_Q': op_settings['b_in_Q'],
                       'l_in_Q': op_settings['l_in_Q'],
                       'p_in_Q': op_settings['p_in_Q']}
        op_temp['Q_' + NR_name] = op_q_spherical.spherical_order_parameter(
            coord, direct, box_length, setting, n_list, thread_num)
        t_end = time.time()
        print("# Spherical Q elap time ", t_end - t_start)

    # Wigner Order parameter W
    if 'W' in op_settings['analysis_type']:
        t_start = time.time()
        setting = {'ave_times': op_settings['ave_times'],
                       'oi_oj': op_settings['oi_oj'],
                       'o_factor': op_settings['o_factor'],
                       'b_in_Q': op_settings['b_in_Q'],
                       'l_in_Q': op_settings['l_in_Q'],
                       'p_in_Q': op_settings['p_in_Q']}
        op_temp['W_' + NR_name] = op_w_wigner.w_order_parameter(
            coord, direct, box_length, setting, n_list, thread_num)
        t_end = time.time()
        print("# Wigner W elap time ", t_end - t_start)

    # Onsager Order parameter S
    if 'S' in op_settings['analysis_type']:
        t_start = time.time()
        setting = {'ave_times' : op_settings['ave_times'],
                        'n_in_S' : op_settings['n_in_S']}
        op_temp['S_' + NR_name] = op_s_local_onsager.onsager_order_parameter(
            direct, setting, n_list, thread_num)
        t_end = time.time()
        print("# Onsager S elap time ", t_end - t_start)

    # McMillan Order parameter T
    if 'T' in op_settings['analysis_type']:
        t_start = time.time()
        setting = {'ave_times': op_settings['ave_times'],
                       'oi_oj': op_settings['oi_oj'],
                       'o_factor': op_settings['o_factor'],
                       'n_in_T' : op_settings['n_in_T'],
                       'd_in_T': op_settings['d_in_T']}
        op_temp['T_' + NR_name] = op_t_msigma.mcmillan_order_parameter(
            coord, direct, box_length, setting, n_list, thread_num)
        t_end = time.time()
        print("# McMillan T elap time ", t_end - t_start)

    # User define order parameter
    if 'Z' in op_settings['analysis_type']:
        t_start = time.time()
        op_temp['Z_' + NR_name] = op_z_user1_define.user1_define_parameter(
            coord, direct, box_length, n_list)
        t_end = time.time()
        print("# User define Z elap time ", t_end - t_start)


    for iname in sorted(op_temp):
        for jname in sorted(op_temp[iname]):
            op_data[iname + '_' + jname] = op_temp[iname][jname]
    
    return op_data

def op_analyze(coord, direct, box_length, op_settings, thread_num):
    """ order parameter analyze
    :param method: method for analysis.  Please see the details in the manual.
    :param coord: = [[0,0,0],[1,0,0]]
    :param direct: = direction vector [[1,0,0],[1,0,0]] or quaternion [[1,0,0,0], [1,0,0,0]] or [] for no direction vector particle
    :param box_length: = [10,10,10]
    :param op_settings: settings for calculating order parameters
    :param thread_num:
    :return op_data:

    """

    coord = np.array(coord)
    direct = np.array(direct)

    # init direct
    if len(direct) == 0:
        direct = [[1,0,0] for i in range(len(coord))]
    if len(direct[0]) == 4:
        direct = misc.q_to_xyz(direct)
    direct = misc.vec_to_unit_vec(direct)

    # calc initial thresh distance
    if 'neighbor' not in op_settings and 'radius' not in op_settings:
        print("set neighborhood param like 'neighbor' or radius'")
    if 'neighbor' not in op_settings:
        op_settings['neighbor'] = []

    dist = 0.0
    if op_settings['neighbor'] != []:
        safe_factor = 1.7
        dist = neighbor_build.calc_thresh(
            box_length, len(coord), max(op_settings['neighbor']), safe_factor)
    if 'radius' not in op_settings:
        op_settings['radius'] = []
    dist = max(op_settings['radius'] + [dist])

    # calc initial neighbor list
    [nei_list, nei_dist] = neighbor_build.build_neighbor_list(
        coord, box_length, {'mode': 'thresh', 'dist': dist}, thread_num)

    # cut up neighbor_list for small radius or num of neighbor.
    neighbors = {}
    for i in op_settings['neighbor']:
        done = False
        while done is False:
            try:
                neighbors['N' + str(i)] = neighbor_build.mod_neighbor_list(
                    nei_list, nei_dist, i, 0)
            except neighbor_build.SmallRadiusError:
                # cutting up failed
                dist = safe_factor * dist
                [nei_list, nei_dist] = neighbor_build.build_neighbor_list(
                    coord, box_length, {'mode': 'thresh', 'dist': dist}, thread_num)
            else:
                done = True

    for i in op_settings['radius']:
        name = ('%03.2f' % i)
        neighbors['R' + name] = neighbor_build.mod_neighbor_list(
            nei_list, nei_dist, 0, i)

    # analyze
    op_data = {}
    for NR_name in sorted(neighbors):
        n_list = neighbors[NR_name][0]
        op_data = op_analyze_with_neighbor_list(
            coord, direct, box_length, NR_name, op_settings, n_list, op_data, thread_num)

    return op_data
