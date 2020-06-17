import numpy as np

from grabcut.jf_bfs.jf_bfs_seq.cfg import *


def count_active_block(node_excess, node_label_height):
    img_shape = node_excess.shape
    block_size_h = (img_shape[0] - 1) // BLOCK_SIZE + 1
    block_size_w = (img_shape[1] - 1) // BLOCK_SIZE + 1
    block_shape = (block_size_h, block_size_w)
    block_size_1D = block_size_h * block_size_w

    active_block = np.zeros(block_size_1D + 1, dtype=int)
    active_block_odd = np.zeros(block_size_1D + 1, dtype=int)

    for block_i in range(block_shape[0]):
        for block_j in range(block_shape[1]):
            count_active_block_deal_each(node_excess, node_label_height,
                                         block_i, block_j, active_block, active_block_odd, block_shape)

    for i in range(1, active_block.shape[0]):
        active_block[i] += active_block[i - 1]
    for i in range(1, active_block_odd.shape[0]):
        active_block_odd[i] += active_block_odd[i - 1]

    list_size = active_block[-1]
    list_size_odd = active_block_odd[-1]

    block_offset = np.array([0, list_size, list_size_odd + list_size])

    list_data = np.zeros((block_size_1D, 2), dtype=int)
    scatter_node(list_data, list_size, active_block, active_block_odd, block_shape)

    return list_data, block_offset


def count_active_block_deal_each(node_excess, node_label_height, block_i, block_j, active_block, active_block_odd,
                                 block_shape):
    offset_i = block_i * BLOCK_SIZE
    offset_j = block_j * BLOCK_SIZE

    for node_i in range(offset_i, offset_i + BLOCK_SIZE):
        for node_j in range(offset_j, offset_j + BLOCK_SIZE):
            if node_excess[node_i, node_j] >= 0 and node_label_height[node_i, node_j] < MAX:
                gid = block_i * block_shape[1] + block_j
                parity = (block_i + block_j) & 1
                active_block[gid + 1] = (1 - parity)
                active_block_odd[gid + 1] = parity


def scatter_node(list_data, offset, active_block, active_block_odd, block_shape):
    for block_i in range(block_shape[0]):
        for block_j in range(block_shape[1]):
            gid = block_i * block_shape[1] + block_j

            prev = active_block[gid]
            current = active_block[gid + 1]
            if current > prev:
                list_data[prev] = [block_i, block_j]

            prev = active_block_odd[gid]
            current = active_block_odd[gid + 1]
            if current > prev:
                list_data[offset + prev] = [block_i, block_j]