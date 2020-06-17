import numpy as np

from grabcut.jf_bfs.jf_bfs_seq.cfg import *

"""
根据bfs得到的block优先级顺序，做全局的relabel（其实是block级别的全局，实际block内部还是无序）
"""


def global_relabel_jf_init(block_nearest_yx,
                           block_nearest_distance,
                           node_excess):
    img_shape = node_excess.shape
    block_shape = block_nearest_yx.shape
    for block_i in range(block_shape[0]):
        for block_j in range(block_shape[1]):
            global_relabel_jf_init_each_block(block_nearest_yx, block_nearest_distance, node_excess, img_shape,
                                              block_i, block_j)


def global_relabel_jf_init_each_block(block_nearest_yx,
                                      block_nearest_distance,
                                      node_excess,
                                      img_shape,
                                      block_i,
                                      block_j):
    offset_i = block_i * BLOCK_SIZE
    offset_j = block_j * BLOCK_SIZE

    for node_i in range(offset_i, offset_i + BLOCK_SIZE):
        if node_i >= img_shape[0]:
            break
        for node_j in range(offset_j, offset_j + BLOCK_SIZE):
            if node_j >= img_shape[1]:
                break
            if node_excess[node_i, node_j] < 0:
                block_nearest_yx[block_i, block_j] = [block_i, block_j]
                block_nearest_distance[block_i, block_j] = 0
                return

    block_nearest_yx[block_i, block_j] = [MAX, MAX]
    block_nearest_distance[block_i, block_j] = MAX


def global_relabel(node_excess, node_capacity, list_data, histogram_data, max_distance):
    node_height = np.where(node_excess < 0, 0, MAX)
    img_shape = node_excess.shape

    done = np.zeros(1, dtype=bool)
    _from, _end, _step = (0, max_distance, 1)
    while not done[0]:
        done[0] = True
        for i in range(_from, _end, _step):
            start = histogram_data[i]
            end = histogram_data[i + 1]
            if start == end:
                continue
            global_relabel_each_distance(list_data, node_height, node_capacity, img_shape, start, end, done)

        _from, _end, _step = (_end - _step, _from - _step, -_step)

    return node_height


def global_relabel_each_distance(list_data, node_height, node_capacity, img_shape, start, end, done):
    for i in range(start, end):
        global_relabel_each_block(list_data[i], node_height, node_capacity, img_shape, done)


def global_relabel_each_block(block, node_height, node_capacity, img_shape, done):
    block_i, block_j = block
    offset_i = block_i * BLOCK_SIZE
    offset_j = block_j * BLOCK_SIZE

    shared_node_height = np.zeros((BLOCK_SIZE + 2, BLOCK_SIZE + 2), dtype=int)
    for node_i in range(offset_i, offset_i + BLOCK_SIZE):
        for node_j in range(offset_j, offset_j + BLOCK_SIZE):
            if node_i >= img_shape[0] or node_j >= img_shape[1]:
                break
            shared_i, shared_j = node_i - offset_i, node_j - offset_j
            shared_node_height[shared_i + 1, shared_j + 1] = node_height[node_i, node_j]
            if node_height[node_i, node_j] > 0:
                if node_i == offset_i:  # up
                    shared_node_height[shared_i, shared_j + 1] = node_height[node_i - 1, node_j] if node_i > 0 else 0
                if node_j == offset_j:  # left
                    shared_node_height[shared_i + 1, shared_j] = node_height[node_i, node_j - 1] if node_j > 0 else 0
                if node_i == min(offset_i + BLOCK_SIZE, img_shape[0]) - 1:  # down
                    shared_node_height[shared_i + 2, shared_j + 1] = (
                        node_height[node_i + 1, node_j] if node_i < img_shape[0] - 1 else 0)
                if node_j == min(offset_j + BLOCK_SIZE, img_shape[1]) - 1:  # right
                    shared_node_height[shared_i + 1, shared_j + 2] = (
                        node_height[node_i, node_j + 1] if node_j < img_shape[1] - 1 else 0)

    block_done = False
    global_done = True
    while not block_done:
        block_done = True
        for node_i in range(offset_i, offset_i + BLOCK_SIZE):
            for node_j in range(offset_j, offset_j + BLOCK_SIZE):
                if node_i >= img_shape[0] or node_j >= img_shape[1]:
                    break
                if node_height[node_i, node_j] > 0:
                    shared_i, shared_j = node_i - offset_i, node_j - offset_j
                    c = node_capacity[node_i, node_j]
                    min_h = node_height[node_i, node_j]
                    if c[0] > 0:  # up
                        min_h = min(min_h, shared_node_height[shared_i, shared_j + 1] + 1)
                    if c[1] > 0:  # down
                        min_h = min(min_h, shared_node_height[shared_i + 2, shared_j + 1] + 1)
                    if c[2] > 0:  # left
                        min_h = min(min_h, shared_node_height[shared_i + 1, shared_j] + 1)
                    if c[3] > 0:  # right
                        min_h = min(min_h, shared_node_height[shared_i + 1, shared_j + 2] + 1)

                    if min_h != node_height[node_i, node_j]:
                        node_height[node_i, node_j] = min_h
                        block_done = False
                        global_done = False
    if not global_done:
        print(np.sum(node_height == 4294967295))
        done[0] = False
