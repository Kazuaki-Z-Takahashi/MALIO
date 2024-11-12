# -*- coding: utf-8 -*-

import numpy as np


def user4_define_parameter(coord, direct, box_length, neighbor_list):

    data_list = []
    for i_i in range(len(coord)):
        sum_distance = 0
        for i_j in range(len(neighbor_list[i_i])):
            x_i_j = [coord[i_i][i] - coord[i_j][i] for i in range(3)]
            sum_distance += np.linalg.norm(x_i_j)
        data_list.append(sum_distance)

    return {'user4_define': data_list }
