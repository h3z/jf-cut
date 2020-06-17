import pickle

import numpy as np

from grabcut.jf_bfs.jf_bfs_seq.cfg import *
from grabcut.jf_bfs.jf_bfs_seq.jumpFloodBFS import compute_bfs


def compute_tag_init(block_nearest_yx, block_nearest_distance, node_excess, tag_data, block_shape):
    for block_i in range(block_shape[0]):
        for block_j in range(block_shape[1]):
            compute_tag_init_each_block(block_nearest_yx, block_nearest_distance, node_excess, tag_data, block_i,
                                        block_j)


def compute_tag_init_each_block(block_nearest_yx, block_nearest_distance, node_excess, tag_data, block_i, block_j):
    img_shape = node_excess.shape
    offset_i = block_i * BLOCK_SIZE
    offset_j = block_j * BLOCK_SIZE

    block_done = False
    for node_i in range(offset_i, offset_i + BLOCK_SIZE):
        for node_j in range(offset_j, offset_j + BLOCK_SIZE):
            if node_i >= img_shape[0] or node_j >= img_shape[1]:
                break

            if node_excess[node_i, node_j] > 0:
                tag_data[node_i, node_j] = MAX
                block_done = True
            else:
                tag_data[node_i, node_j] = 0

    if block_done:
        block_nearest_distance[block_i, block_j] = 0
        block_nearest_yx[block_i, block_j] = [block_i, block_j]
    else:
        block_nearest_yx[block_i, block_j] = [MAX, MAX]
        block_nearest_distance[block_i, block_j] = MAX


def compute_tag(block_nearest_yx, block_nearest_distance, node_excess, cut_data, tag_data, block_shape):
    compute_tag_init(block_nearest_yx, block_nearest_distance, node_excess, tag_data, block_shape)
    list_data, histogram_data, max_distance = compute_bfs(block_nearest_yx, block_nearest_distance, block_shape)

    pickle.dump(block_nearest_distance, open('/home/hzzz/jiubai_env/data/tag_distance.p', 'wb'))
    done = np.array([False])
    _from, _end, _step = (0, max_distance, 1)
    while not done[0]:
        done[:] = [True]
        for i in range(_from, _end, _step):
            start = histogram_data[i]
            end = histogram_data[i + 1]
            if start == end:
                continue
            compute_tag_each_distance(list_data, tag_data, cut_data, start, end, done)

        _from, _end, _step = (_end - _step, _from - _step, -_step)


def compute_tag_each_block(block, tag_data, cut_data, img_shape, done):
    block_i, block_j = block
    offset_i = block_i * BLOCK_SIZE
    offset_j = block_j * BLOCK_SIZE

    shared_tag_data = np.zeros((BLOCK_SIZE + 2, BLOCK_SIZE + 2), dtype=int)
    for node_i in range(offset_i, offset_i + BLOCK_SIZE):
        for node_j in range(offset_j, offset_j + BLOCK_SIZE):
            if node_i >= img_shape[0] or node_j >= img_shape[1]:
                break
            shared_i, shared_j = node_i - offset_i, node_j - offset_j
            shared_tag_data[shared_i + 1, shared_j + 1] = tag_data[node_i, node_j]
            if cut_data[node_i, node_j] == MAX and tag_data[node_i, node_j] == 0:
                if node_i == offset_i:  # up
                    if node_i > 0:
                        shared_tag_data[shared_i + 0, shared_j + 1] = tag_data[node_i - 1, node_j]
                    else:
                        shared_tag_data[shared_i + 0, shared_j + 1] = 0
                if node_j == offset_j:  # left
                    if node_j > 0:
                        shared_tag_data[shared_i + 1, shared_j + 0] = tag_data[node_i, node_j - 1]
                    else:
                        shared_tag_data[shared_i + 1, shared_j + 0] = 0
                if node_i == min(offset_i + BLOCK_SIZE, img_shape[0]) - 1:  # down
                    if node_i < img_shape[0] - 1:
                        shared_tag_data[shared_i + 2, shared_j + 1] = tag_data[node_i + 1, node_j]
                    else:
                        shared_tag_data[shared_i + 2, shared_j + 1] = 0
                if node_j == min(offset_j + BLOCK_SIZE, img_shape[1]) - 1:  # right
                    if node_j < img_shape[1] - 1:
                        shared_tag_data[shared_i + 1, shared_j + 2] = tag_data[node_i, node_j + 1]
                    else:
                        shared_tag_data[shared_i + 1, shared_j + 2] = 0

    block_done = False
    while not block_done:
        block_done = True
        for node_i in range(offset_i, offset_i + BLOCK_SIZE):
            for node_j in range(offset_j, offset_j + BLOCK_SIZE):
                if node_i >= img_shape[0] or node_j >= img_shape[1]:
                    break
                shared_i, shared_j = node_i - offset_i, node_j - offset_j
                c = shared_tag_data[shared_i + 1, shared_j + 1]
                if cut_data[node_i, node_j] and c == 0:
                    if shared_tag_data[shared_i, shared_j + 1]:
                        c = MAX
                    elif shared_tag_data[shared_i + 2, shared_j + 1]:
                        c = MAX
                    elif shared_tag_data[shared_i + 1, shared_j]:
                        c = MAX
                    elif shared_tag_data[shared_i + 1, shared_j + 2]:
                        c = MAX
                    if c == MAX:
                        tag_data[node_i, node_j] = c
                        shared_tag_data[shared_i + 1, shared_j + 1] = c
                        block_done = False
                        done[0] = False


def compute_tag_each_distance(list_data, tag_data, cut_data, start, end, done):
    img_shape = tag_data.shape
    for i in range(start, end):
        compute_tag_each_block(list_data[i], tag_data, cut_data, img_shape, done)
