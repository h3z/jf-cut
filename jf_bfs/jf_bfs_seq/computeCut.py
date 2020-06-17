import numpy as np

from grabcut.jf_bfs.jf_bfs_seq.cfg import *
from grabcut.jf_bfs.jf_bfs_seq.jumpFloodBFS import compute_bfs


def compute_cut_init(block_nearest_yx, block_nearest_distance, node_excess, cut_data, block_shape):
    for block_i in range(block_shape[0]):
        for block_j in range(block_shape[1]):
            compute_cut_init_each_block(block_nearest_yx, block_nearest_distance, node_excess, cut_data, block_i,
                                        block_j)


def compute_cut_init_each_block(block_nearest_yx, block_nearest_distance, node_excess, cut_data, block_i, block_j):
    img_shape = node_excess.shape
    offset_i = block_i * BLOCK_SIZE
    offset_j = block_j * BLOCK_SIZE

    block_done = False
    for node_i in range(offset_i, offset_i + BLOCK_SIZE):
        for node_j in range(offset_j, offset_j + BLOCK_SIZE):
            if node_i >= img_shape[0] or node_j >= img_shape[1]:
                break

            if node_excess[node_i, node_j] > 0:
                cut_data[node_i, node_j] = MAX
                block_done = True
            else:
                cut_data[node_i, node_j] = 0

    if block_done:
        block_nearest_distance[block_i, block_j] = 0
        block_nearest_yx[block_i, block_j] = [block_i, block_j]
    else:
        block_nearest_yx[block_i, block_j] = [MAX, MAX]
        block_nearest_distance[block_i, block_j] = MAX


def compute_cut(block_nearest_yx, block_nearest_distance, node_capacity, node_excess, cut_data, block_shape):
    compute_cut_init(block_nearest_yx, block_nearest_distance, node_excess, cut_data, block_shape)
    list_data, histogram_data, max_distance = compute_bfs(block_nearest_yx, block_nearest_distance, block_shape)

    done = np.array([False, False])
    _from, _end, _step = (0, max_distance, 1)
    while not done[0] and not done[1]:
        done[:] = [True, False]
        for i in range(_from, _end, _step):
            start = histogram_data[i]
            end = histogram_data[i + 1]
            if start == end:
                continue
            compute_cut_each_distance(list_data, cut_data, node_capacity, node_excess, start, end, done)

        _from, _end, _step = (_end - _step, _from - _step, -_step)
    return not done[1]


def compute_cut_each_block(block, cut______data, node_capacity, node_excess, img_shape, done):
    block_i, block_j = block
    offset_i = block_i * BLOCK_SIZE
    offset_j = block_j * BLOCK_SIZE

    block_done = done[1]
    if block_done:
        return

    shared_cut_data = np.zeros((BLOCK_SIZE + 2, BLOCK_SIZE + 2), dtype=int)
    shared_capacity_from = np.zeros((BLOCK_SIZE, BLOCK_SIZE, 4))
    for node_i in range(offset_i, offset_i + BLOCK_SIZE):
        for node_j in range(offset_j, offset_j + BLOCK_SIZE):
            if node_i >= img_shape[0] or node_j >= img_shape[1]:
                break
            shared_i, shared_j = node_i - offset_i, node_j - offset_j
            shared_cut_data[shared_i + 1, shared_j + 1] = cut______data[node_i, node_j]
            if cut______data[node_i, node_j] == 0:
                if node_i == offset_i:  # up
                    if node_i > 0:
                        shared_cut_data[shared_i + 0, shared_j + 1] = cut______data[node_i - 1, node_j]
                    else:
                        shared_cut_data[shared_i + 0, shared_j + 1] = 0
                if node_j == offset_j:  # left
                    if node_j > 0:
                        shared_cut_data[shared_i + 1, shared_j + 0] = cut______data[node_i, node_j - 1]
                    else:
                        shared_cut_data[shared_i + 1, shared_j + 0] = 0
                if node_i == min(offset_i + BLOCK_SIZE, img_shape[0]) - 1:  # down
                    if node_i < img_shape[0] - 1:
                        shared_cut_data[shared_i + 2, shared_j + 1] = cut______data[node_i + 1, node_j]
                    else:
                        shared_cut_data[shared_i + 2, shared_j + 1] = 0
                if node_j == min(offset_j + BLOCK_SIZE, img_shape[1]) - 1:  # right
                    if node_j < img_shape[1] - 1:
                        shared_cut_data[shared_i + 1, shared_j + 2] = cut______data[node_i, node_j + 1]
                    else:
                        shared_cut_data[shared_i + 1, shared_j + 2] = 0

                if node_i > 0:
                    shared_capacity_from[shared_i, shared_j, 0] = node_capacity[node_i - 1, node_j, 1]
                else:
                    shared_capacity_from[shared_i, shared_j, 0] = 0
                if node_i < img_shape[0] - 1:
                    shared_capacity_from[shared_i, shared_j, 1] = node_capacity[node_i + 1, node_j, 0]
                else:
                    shared_capacity_from[shared_i, shared_j, 1] = 0
                if node_j > 0:
                    shared_capacity_from[shared_i, shared_j, 2] = node_capacity[node_i, node_j - 1, 3]
                else:
                    shared_capacity_from[shared_i, shared_j, 2] = 0
                if node_j < img_shape[1] - 1:
                    shared_capacity_from[shared_i, shared_j, 3] = node_capacity[node_i, node_j + 1, 2]
                else:
                    shared_capacity_from[shared_i, shared_j, 3] = 0

    while not block_done:
        block_done = True
        for node_i in range(offset_i, offset_i + BLOCK_SIZE):
            for node_j in range(offset_j, offset_j + BLOCK_SIZE):
                if node_i >= img_shape[0] or node_j >= img_shape[1]:
                    break
                shared_i, shared_j = node_i - offset_i, node_j - offset_j
                c = shared_cut_data[shared_i + 1, shared_j + 1]
                if c == 0:
                    capacity = shared_capacity_from[shared_i, shared_j]
                    if capacity[0] > 0 and shared_cut_data[shared_i, shared_j + 1]:
                        c = MAX
                    elif capacity[1] > 0 and shared_cut_data[shared_i + 2, shared_j + 1]:
                        c = MAX
                    elif capacity[2] > 0 and shared_cut_data[shared_i + 1, shared_j]:
                        c = MAX
                    elif capacity[3] > 0 and shared_cut_data[shared_i + 1, shared_j + 2]:
                        c = MAX
                    if c == MAX:
                        cut______data[node_i, node_j] = c
                        shared_cut_data[shared_i + 1, shared_j + 1] = c
                        if node_excess[node_i, node_j] < 0:
                            done[1] = True
                        block_done = False
                        done[0] = False


def compute_cut_each_distance(list_data, cut_data, node_capacity, node_excess, start, end, done):
    img_shape = node_capacity.shape[:2]
    for i in range(start, end):
        compute_cut_each_block(list_data[i], cut_data, node_capacity, node_excess, img_shape, done)
        if done[1]:
            return
