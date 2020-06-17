import numpy as np

from grabcut.jf_bfs.jf_bfs_seq.cfg import *

"""
function jump_flood:
    将形状位为node_excess.shape的图，分割成shape / BLOCK_SIZE份。
    残余小于0为背景点，以包含背景点的BLOCK为起始，jump flood方法扩散开，
    返回扩散后的BLOCK情况，即：每个BLOCK，与之最近的BLOCK是谁&距离是多少
    
function compute_bfs:
    在那些block的距离的基础上，根据距离排列优先级，返回一个TODO列表
"""


def jump_flood(block_nearest_yx, block_nearest_distance, block_shape):

    jump_size = 1
    max_jump_size = max(block_shape[0], block_shape[1])
    while jump_size < max_jump_size:
        jump_size <<= 1

    while jump_size != 1:
        jump_size >>= 1
        jump_flooding(block_nearest_yx, block_nearest_distance, block_shape, jump_size)
        jump_flooding(block_nearest_yx, block_nearest_distance, block_shape, jump_size)

    return block_nearest_yx, block_nearest_distance


def jump_flooding(block_nearest_yx,
                  block_nearest_distance,
                  block_shape,
                  jump_size):
    for block_i in range(block_shape[0]):
        for block_j in range(block_shape[1]):
            jump_flooding_each_block(block_nearest_yx, block_nearest_distance, block_shape, jump_size, block_i, block_j)


def jump_flooding_each_block(block_nearest_yx,
                             block_nearest_distance,
                             block_shape,
                             jump_size,
                             block_i,
                             block_j):
    right = (block_i, block_j + jump_size)
    left = (block_i, block_j - jump_size)
    up = (block_i - jump_size, block_j)
    down = (block_i + jump_size, block_j)

    min_distance = block_nearest_distance[block_i, block_j]
    min_nearest = block_nearest_yx[block_i, block_j]
    if up[0] >= 0 and block_nearest_distance[up] != MAX:
        nearest = block_nearest_yx[up]
        distance = get_distance(block_i, block_j, nearest)
        if distance < min_distance:
            min_distance = distance
            min_nearest = nearest

    if down[0] < block_shape[0] and block_nearest_distance[down] != MAX:
        nearest = block_nearest_yx[down]
        distance = get_distance(block_i, block_j, nearest)
        if distance < min_distance:
            min_distance = distance
            min_nearest = nearest

    if right[1] < block_shape[1] and block_nearest_distance[right] != MAX:
        nearest = block_nearest_yx[right]
        distance = get_distance(block_i, block_j, nearest)
        if distance < min_distance:
            min_distance = distance
            min_nearest = nearest

    if left[1] >= 0 and block_nearest_distance[left] != MAX:
        nearest = block_nearest_yx[left]
        distance = get_distance(block_i, block_j, nearest)
        if distance < min_distance:
            min_distance = distance
            min_nearest = nearest

    if min_distance != MAX:
        block_nearest_distance[block_i, block_j] = min_distance
        block_nearest_yx[block_i, block_j] = min_nearest


def get_distance(block_i, block_j, nearest):
    i, j = nearest
    return abs(block_i - i) + abs(block_j - j)


def compute_bfs(block_nearest_yx, block_nearest_distance, block_shape):
    jump_flood(block_nearest_yx, block_nearest_distance, block_shape)

    histogram_data = np.zeros(1 + block_shape[0] + block_shape[1], dtype=int)  # 最大距离是 width + height
    count_groupby = np.zeros(block_shape, dtype=int)
    count_groupby.fill(MAX)  # useless, JUST FOR DEBUG

    # group by & order
    max_distance = 0
    for block_i in range(block_shape[0]):
        for block_j in range(block_shape[1]):
            distance = block_nearest_distance[block_i, block_j]
            count_groupby[block_i, block_j] = histogram_data[distance + 1]
            histogram_data[distance + 1] += 1
            max_distance = max(max_distance, distance + 1)

    # prefix sum
    for i in range(max_distance):
        histogram_data[i + 1] += histogram_data[i]

    list_data = np.zeros((block_shape[0] * block_shape[1], 2), dtype=int)
    for block_i in range(block_shape[0]):
        for block_j in range(block_shape[1]):
            distance = block_nearest_distance[block_i, block_j]
            position = histogram_data[distance] + count_groupby[block_i, block_j]
            list_data[position] = [block_i, block_j]
    return list_data, histogram_data, max_distance
