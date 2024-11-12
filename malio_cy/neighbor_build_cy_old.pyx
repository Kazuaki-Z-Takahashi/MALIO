# -*- coding: utf-8 -*-

import time
from multiprocessing import Pool, Array
import sys
import numpy as np
import misc_cy
from malio_cy_def cimport *
cimport cython

class SmallRadiusError(Exception):

    def __str__(self):
        return 'neighbor radius too small'


def calc_thresh(box_length, part_num, target_num, safe_factor):
    density = float(part_num) / (box_length[0] * box_length[1] * box_length[2])
    target_r = (target_num / (density * (4.0 / 3.0) * math.pi))**(1.0 / 3.0)

    return safe_factor * target_r


def build_neighbor_cell(cell, cell_size):
    # input  : [0,0,0]
    # output : if cell_size == [4,4,4] , [-1,0,0] => [3,0,0], [4,0,0] =>
    # [0,0,0]
    neighbor = [[cell[0] + ix, cell[1] + iy, cell[2] + iz]
                for ix in range(-1, 2)
                for iy in range(-1, 2)
                for iz in range(-1, 2)]
    for i in range(3**3):
        for j in range(3):
            if neighbor[i][j] == -1:
                neighbor[i][j] = cell_size[j] - 1
            elif neighbor[i][j] == cell_size[j]:
                neighbor[i][j] = 0
    return neighbor

def wrapper_cell_calc(args):
    [i_i, cell_length, cell_size, cell_list, box_length, condition] = args

    nei_list = []
    nei_dist = []

    coord_ii = [coord_1d[3 * i_i + i] for i in range(3)]

    cell = coord_to_cell_num(coord_ii, cell_length)
    neighbor = build_neighbor_cell(cell, cell_size)
    for inei in neighbor:
        for i_j in cell_list[inei[0]][inei[1]][inei[2]]:
            if i_i < i_j:
                coord_i_j = [coord_1d[3 * i_j + i] for i in range(3)]
                dist = np.linalg.norm(
                    misc_cy.calc_delta(coord_ii, coord_i_j, box_length))
                if condition['mode'] == 'thresh':
                    if dist <= condition['dist']:
                        nei_list.append(i_j)
                        nei_dist.append(dist)
                elif condition['mode'] == 'neighbor':
                    if len(nei_list) < condition['num']:
                        nei_list.append(i_j)
                        nei_dist.append(dist)
                    else:
                        if max(nei_dist) > dist:
                            idx = nei_dist.index(max(nei_dist))
                            nei_list[idx] = i_j
                            nei_dist[idx] = dist
    return [nei_list, nei_dist]


def coord_to_cell_num(coord, cell_length):
    inum = [int(coord[i] / cell_length[i]) for i in range(3)]
    return inum


def add_num_dist(nei_list, nei_dist, num, i_j, dist):
    # add i_j to nei_list and nei_dist
    if len(nei_list) < num:
        nei_list.append(i_j)
        nei_dist.append(dist)
    else:
        if max(nei_dist) > dist:
            idx = nei_dist.index(max(nei_dist))
            nei_list[idx] = i_j
            nei_dist[idx] = dist
    return [nei_list, nei_dist]


def build_cell(coord, box_length, thresh_dist):
    cell_length = [0, 0, 0]
    cell_size = [0, 0, 0]
    for i in range(3):
        cell_size[i] = int(box_length[i] / thresh_dist)
        cell_length[i] = box_length[i] / float(cell_size[i])

    cell_list = \
        [[[[] for _ in range(cell_size[2])] for _ in range(cell_size[1])]
         for _ in range(cell_size[0])]

    for i, _ in enumerate(coord):
        # periodic boundary condition check
        for j in range(3):
            if coord[i][j] < 0.0:
                coord[i][j] += box_length[j]
            if coord[i][j] >= box_length[j]:
                coord[i][j] -= box_length[j]

        inum = coord_to_cell_num(coord[i], cell_length)
        cell_list[inum[0]][inum[1]][inum[2]].append(i)

    return [cell_list, cell_length, cell_size]


def build_neighbor_list_org(coord, box_length, condition, thread_num):
    """ building neighbor list
    :param coord: = [[0,0,0],[1,0,0]]
    :param box_length: = [10,10,10]
    :param condition: =
    {'mode' : 'thresh' or 'neighbor',
        'neighbor' is number of neighborhood particles,
        'dist' : is radii of neighborhood particles. }
    :return [neighbor_list, neidhbor_distance]: is new neighbor list
    """
    t_start = time.time()
    [cell_list, cell_length, cell_size] = build_cell(
        coord, box_length, condition['dist'])

    if min(cell_size) <= 2:
        nei_list = [[] for i in range(len(coord))]
        nei_dist = [[] for i in range(len(coord))]
        for i_i in range(len(coord) - 1):
            for i_j in range(i_i + 1, len(coord)):
                dist = np.linalg.norm(
                    misc_cy.calc_delta(coord[i_i], coord[i_j], box_length))
                if condition['mode'] == 'thresh':
                    if dist <= condition['dist']:
                        nei_list[i_i].append(i_j)
                        nei_dist[i_i].append(dist)
                        nei_list[i_j].append(i_i)
                        nei_dist[i_j].append(dist)
                elif condition['mode'] == 'neighbor':
                    [nei_list[i_i], nei_dist[i_i]] = \
                        add_num_dist(
                            nei_list[i_i], nei_dist[i_i], condition['num'], i_j, dist)
                    [nei_list[i_j], nei_dist[i_j]] = \
                        add_num_dist(
                            nei_list[i_j], nei_dist[i_j], condition['num'], i_i, dist)
        [nei_list, nei_dist] = misc_cy.sort_by_distance(nei_list, nei_dist)
        return [nei_list, nei_dist]

    # prepare parallel
    global coord_1d
    coord_1d = Array('d', misc_cy.convert_3dim_to_1dim(coord), lock=False)

    ### now_pool = Pool(thread_num)
    args = [[i, cell_length, cell_size, cell_list, box_length, condition]
            for i in range(len(coord))]
    ### out_data = now_pool.map(wrapper_cell_calc, args)
    out_data = []
    for arg in args:
        res = wrapper_cell_calc(arg)
        out_data.append(res)
    ### now_pool.close()

    del coord_1d

    nei_list = [[] for i in range(len(coord))]
    nei_dist = [[] for i in range(len(coord))]
    for i in range(len(coord)):
        nei_list[i] = out_data[i][0]
        nei_dist[i] = out_data[i][1]

    for i, _ in enumerate(nei_list):
        for j, _ in enumerate(nei_list[i]):
            now_j = nei_list[i][j]
            if i < now_j:
                if condition['mode'] == 'thresh':
                    nei_list[now_j].append(i)
                    nei_dist[now_j].append(nei_dist[i][j])
                elif condition['mode'] == 'neighbor':
                    [nei_list[now_j], nei_dist[now_j]] = \
                        add_num_dist(nei_list[now_j], nei_dist[now_j],
                                     condition['num'], i, nei_dist[i][j])

    [nei_list, nei_dist] = misc_cy.sort_by_distance(nei_list, nei_dist)

    # check
    if condition['mode'] == 'neighbor':
        if len(nei_list) < condition['num']:
            print('# neighbor num too big. you require ',
                  condition['num'], ' neighbors. But there are ', len(nei_list), 'particles.')
            sys.exit(1)
        for i, _ in enumerate(nei_list):
            if len(nei_list[i]) < condition['num']:
                print('# radius too small. you require ', condition['num'],
                      ' neighbors. But there are ', len(nei_list[i]), 'neighbors.')
                sys.exit(1)

    t_end = time.time()
    print("# neighbor elap time ", t_end - t_start)
    return [nei_list, nei_dist]


def build_neighbor_list(coord, box_length, condition, thread_num):
    t_start = time.time()
    [cell_list, cell_length, cell_size] = build_cell(
        coord, box_length, condition['dist'])

    cdef:
        int i, j, k, l, i_i, i_j, len_c = len(coord), 
        int ix, iy, iz, i_k, conditionnum, now_j
        double dist, conditiondist, delta
        ivec iv = ivec()
        dvec dv = dvec()
        bool modethresh = condition['mode'] == 'thresh'
        bool modeneighbor = condition['mode'] == 'neighbor'
        vector[double] dbox_length = box_length
        vector[vector[double]] dcoord = coord
        vector[double] dcell_length = cell_length
        vector[int] icell_size = cell_size
        dvec coord_ii = dvec(3)
        dvec coord_i_j = dvec(3)
        ivec cell = ivec(3)
        int neighbor[27][3]

    if modethresh:
        conditiondist= condition['dist']
    if modeneighbor:
        conditionnum = condition['num']

    if min(cell_size) <= 2:
        nei_list = [[] for i in range(len_c)]
        nei_dist = [[] for i in range(len_c)]
        for i_i in range(len_c - 1):
            for i_j in range(i_i + 1, len_c):
                dist = np.linalg.norm(
                    misc_cy.calc_delta(coord[i_i], coord[i_j], box_length))
                if condition['mode'] == 'thresh':
                    if dist <= condition['dist']:
                        nei_list[i_i].append(i_j)
                        nei_dist[i_i].append(dist)
                        nei_list[i_j].append(i_i)
                        nei_dist[i_j].append(dist)
                elif condition['mode'] == 'neighbor':
                    [nei_list[i_i], nei_dist[i_i]] = \
                        add_num_dist(
                            nei_list[i_i], nei_dist[i_i], condition['num'], i_j, dist)
                    [nei_list[i_j], nei_dist[i_j]] = \
                        add_num_dist(
                            nei_list[i_j], nei_dist[i_j], condition['num'], i_i, dist)
        [nei_list, nei_dist] = misc_cy.sort_by_distance(nei_list, nei_dist)
        return [nei_list, nei_dist]

    # prepare parallel
    ### global coord_1d
    ### coord_1d = Array('d', misc_cy.convert_3dim_to_1dim(coord), lock=False)
    ### [i, cell_length, cell_size, cell_list, box_length, condition]
    ### [i_i, cell_length, cell_size, cell_list, box_length, condition]

    out_data= []
    for i_i in range(len_c):

        nei_list_1 = []
        nei_dist_1 = []
        ### coord_ii = [coord_1d[3 * i_i + i] for i in range(3)]
        for i in range(3):
            coord_ii[i] = dcoord[i_i][i] ### coord_1d[3 * i_i + i]

        ### cell = coord_to_cell_num(coord_ii, cell_length)
        for i in range(3):
            cell[i] = <int>(coord_ii[i]/dcell_length[i])
        i = 0
        for ix in range(-1, 2):
            for iy in range(-1, 2):
                for iz in range(-1, 2):
                    neighbor[i][0] = cell[0] + ix
                    neighbor[i][1] = cell[1] + iy
                    neighbor[i][2] = cell[2] + iz
                    i += 1
        for i in range(3**3):
            for j in range(3):
                if neighbor[i][j] == -1:
                    neighbor[i][j] = icell_size[j] - 1
                elif neighbor[i][j] == icell_size[j]:
                    neighbor[i][j] = 0
        ### for inei in neighbor:
        for i_k in range(27):
            ### for i_j in cell_list[inei[0]][inei[1]][inei[2]]:
            for i_j in cell_list[neighbor[i_k][0]][neighbor[i_k][1]][neighbor[i_k][2]]:
                if i_i < i_j:
                    ### coord_i_j = [coord_1d[3 * i_j + i] for i in range(3)]
                    for i in range(3):
                        coord_i_j[i] = dcoord[i_j][i]
                    ### dist = np.linalg.norm(
                    ###      misc_cy.calc_delta(coord_ii, coord_i_j, box_length))
                    dist = 0.0
                    for i in range(3):
                        delta = dcoord[i_i][i] - dcoord[i_j][i]
                        if delta < -dbox_length[i] * 0.5:
                            delta += dbox_length[i]
                        elif delta >= dbox_length[i] * 0.5:
                            delta -= dbox_length[i]
                        dist += delta*delta
                    dist = math.sqrt(dist)
                    ### if condition['mode'] == 'thresh':
                    if modethresh:
                        ### if dist <= condition['dist']:
                        if dist <= conditiondist:
                            nei_list_1.append(i_j)
                            nei_dist_1.append(dist)
                    ### elif condition['mode'] == 'neighbor':
                    elif modeneighbor:
                        ### if len(nei_list) < condition['num']:
                        if len(nei_list_1) < conditionnum:
                            nei_list_1.append(i_j)
                            nei_dist_1.append(dist)
                        else:
                            if max(nei_dist_1) > dist:
                                idx = nei_dist_1.index(max(nei_dist_1))
                                nei_list_1[idx] = i_j
                                nei_dist_1[idx] = dist
            res = [nei_list_1, nei_dist_1]

        out_data.append(res)

    ### out_data= []
    ### for arg in args:
    ###     res = wrapper_cell_calc(arg)
    ###     out_data.append(res)
    ### del coord_1d

    nei_list = [[] for i in range(len_c)]
    nei_dist = [[] for i in range(len_c)]
    for i in range(len_c):
        nei_list[i] = out_data[i][0]
        nei_dist[i] = out_data[i][1]     

    ### for i, _ in enumerate(nei_list):
    ###     for j, _ in enumerate(nei_list[i]):
    for i in range(len(nei_list)):
        for j in range(len(nei_list[i])):
            now_j = nei_list[i][j]
            if i < now_j:
                ### if condition['mode'] == 'thresh':
                if modethresh:
                    nei_list[now_j].append(i)
                    nei_dist[now_j].append(nei_dist[i][j])
                elif condition['mode'] == 'neighbor':
                    [nei_list[now_j], nei_dist[now_j]] = \
                        add_num_dist(nei_list[now_j], nei_dist[now_j],
                                     condition['num'], i, nei_dist[i][j])

    [nei_list, nei_dist] = misc_cy.sort_by_distance(nei_list, nei_dist)

    # check
    if condition['mode'] == 'neighbor':
        if len(nei_list) < condition['num']:
            print('# neighbor num too big. you require ',
                  condition['num'], ' neighbors. But there are ', len(nei_list), 'particles.')
            sys.exit(1)
        for i, _ in enumerate(nei_list):
            if len(nei_list[i]) < condition['num']:
                print('# radius too small. you require ', condition['num'],
                      ' neighbors. But there are ', len(nei_list[i]), 'neighbors.')
                sys.exit(1)

    t_end = time.time()
    print("# neighbor elap time ", t_end - t_start)
    return [nei_list, nei_dist]

def build_neighbor_list_old(coord, box_length, condition, int thread_num):
    """ building neighbor list
    :param coord: = [[0,0,0],[1,0,0]]
    :param box_length: = [10,10,10]
    :param condition: =
    {'mode' : 'thresh' or 'neighbor',
        'neighbor' is number of neighborhood particles,
        'dist' : is radii of neighborhood particles. }
    :return [neighbor_list, neidhbor_distance]: is new neighbor list
    """

    cdef:
        int i, j, i_i, i_j, len_c = len(coord)
        double dist, conditiondist, delta
        ivec iv = ivec()
        dvec dv = dvec()
        ivec_vec nei_list = ivec_vec(len_c, iv)
        dvec_vec nei_dist = dvec_vec(len_c, dv)
        bool modethresh = condition['mode'] == 'thresh'
        bool modeneighbor = condition['mode'] == 'neighbor'
        double dbox_length[3]
        double **dcoord = <double**>malloc(len_c*sizeof(double*))

    for i in range(len_c):
        dcoord[i] = <double*>malloc(3*sizeof(double))
    
    if modethresh:
        conditiondist= condition['dist']
    for i in range(len_c):
        for j in range(3):
            dcoord[i][j] = coord[i][j]
    for i in range(3):
        dbox_length[i] = box_length[i]

    t_start = time.time()
    [cell_list, cell_length, cell_size] = build_cell(
        coord, box_length, condition['dist'])

    ### nei_list = [[] for i in range(len(coord))]
    ### nei_dist = [[] for i in range(len(coord))]
    for i_i in range(len_c - 1):
        for i_j in range(i_i + 1, len_c):
            #dist = np.linalg.norm(
            #    misc_cy.calc_delta(coord[i_i], coord[i_j], box_length))
            ### explicit coding for calculating dist
            dist = 0.0
            for i in range(3):
                delta = dcoord[i_i][i] - dcoord[i_j][i]
                if delta < -dbox_length[i] * 0.5:
                    delta += dbox_length[i]
                elif delta >= dbox_length[i] * 0.5:
                    delta -= dbox_length[i]
                dist += delta*delta
            dist = math.sqrt(dist)
            ### if condition['mode'] == 'thresh':
            if modethresh:
                ### if dist <= condition['dist']:
                if dist <= conditiondist:
                    nei_list[i_i].push_back(i_j)
                    nei_dist[i_i].push_back(dist)
                    nei_list[i_j].push_back(i_i)
                    nei_dist[i_j].push_back(dist)
            ### elif condition['mode'] == 'neighbor':
            elif modeneighbor:
                [nei_list[i_i], nei_dist[i_i]] = \
                    add_num_dist(
                        nei_list[i_i], nei_dist[i_i], condition['num'], i_j, dist)
                [nei_list[i_j], nei_dist[i_j]] = \
                    add_num_dist(
                        nei_list[i_j], nei_dist[i_j], condition['num'], i_i, dist)
    [nei_list, nei_dist] = misc_cy.sort_by_distance(nei_list, nei_dist)

    # check
    if condition['mode'] == 'neighbor':
        if len(nei_list) < condition['num']:
            print('# neighbor num too big. you require ',
                  condition['num'], ' neighbors. But there are ', len(nei_list), 'particles.')
            sys.exit(1)
        for i, _ in enumerate(nei_list):
            if len(nei_list[i]) < condition['num']:
                print('# radius too small. you require ', condition['num'],
                      ' neighbors. But there are ', len(nei_list[i]), 'neighbors.')
                sys.exit(1)

    for i in range(len_c):
        free(dcoord[i])
    free(dcoord)

    t_end = time.time()
    print("# neighbor elap time ", t_end - t_start)
    return [nei_list, nei_dist]

def mod_neighbor_list(nei_list, nei_dist, neighbor, radii):
    """ cutting up neighbor_list
    Either A or B must be 0
    :param nei_list: is [[1,2],[0,2],[0,1]
    :param nei_dist: is [[1,2],[1,2],[1,2]]
    :param neighbor: is target number o_f neighbor atoms
    :param radii: is target distance o_f neighbor atoms
    :return [neighbor_list, neighbor_distance]: is cut up neighbor_list
    """
    new_list = [[] for i in range(len(nei_list))]
    new_dist = [[] for i in range(len(nei_list))]

    if neighbor != 0 and radii != 0:
        print("# error in mod neighbor list")
        return []

    if neighbor != 0:
        for i, _ in enumerate(nei_list):
            if len(nei_list[i]) < neighbor:
                print('# radius too small. you require ', neighbor,
                      ' neighbors. But there are ', len(nei_list[i]), 'neighbors.')
                raise SmallRadiusError()
            for j in range(neighbor):
                new_list[i].append(nei_list[i][j])
                new_dist[i].append(nei_dist[i][j])
    elif radii != 0:
        for i, _ in enumerate(nei_list):
            for j in range(len(nei_list[i])):
                dist = nei_dist[i][j]
                if dist <= radii:
                    new_list[i].append(nei_list[i][j])
                    new_dist[i].append(nei_dist[i][j])

    return [new_list, new_dist]
