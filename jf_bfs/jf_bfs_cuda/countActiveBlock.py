import numba
import numpy as np
from numba import cuda

from grabcut.jf_bfs.jf_bfs_cuda.cfg import *


def cuda_count_active_block(node_excess, node_label_height):
    img_shape = node_excess.shape
    block_size_h = (img_shape[0] - 1) // BLOCK_SIZE + 1
    block_size_w = (img_shape[1] - 1) // BLOCK_SIZE + 1
    block_shape = (block_size_h, block_size_w)
    block_size_1D = block_size_h * block_size_w

    active_block = np.zeros(block_size_1D + 1, dtype=int)
    active_block_odd = np.zeros(block_size_1D + 1, dtype=int)

    cuda_count_active_block_deal_each[(block_size_w, block_size_h), (BLOCK_SIZE, BLOCK_SIZE)](
        node_excess, node_label_height, active_block, active_block_odd)

    for i in range(1, active_block.shape[0]):
        active_block[i] += active_block[i - 1]
    for i in range(1, active_block_odd.shape[0]):
        active_block_odd[i] += active_block_odd[i - 1]

    list_size = active_block[-1]
    list_size_odd = active_block_odd[-1]

    block_offset = np.array([0, list_size, list_size_odd + list_size])

    list_data = np.zeros((block_size_1D, 2), dtype=int)
    # scatter_node(list_data, list_size, active_block, active_block_odd, block_shape)
    #  TODO 这里很少，可能还不如串行……
    cuda_scatter_node[(block_size_w, block_size_h), 1](list_data, list_size, active_block, active_block_odd)

    return list_data, block_offset


@cuda.jit
def cuda_count_active_block_deal_each(node_excess, node_label_height, active_block, active_block_odd):
    img_shape = node_excess.shape
    node_j, node_i = cuda.grid(2)
    if node_i >= img_shape[0] or node_j >= img_shape[1]:
        return

    block_done = cuda.shared.array(1, dtype=numba.int8)
    if cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0:
        block_done[0] = 0
    cuda.syncthreads()

    if node_excess[node_i, node_j] >= 0 and node_label_height[node_i, node_j] < MAX:
        block_done[0] = 1
    cuda.syncthreads()

    if cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0:
        block_i, block_j = cuda.blockIdx.y, cuda.blockIdx.x
        gid = block_i * cuda.gridDim.x + block_j
        parity = (block_i + block_j) & 1
        done = block_done[0]
        active_block[gid + 1] = (1 - parity) * done
        active_block_odd[gid + 1] = parity * done


@cuda.jit
def cuda_scatter_node(list_data, offset, active_block, active_block_odd):
    block_i, block_j = cuda.blockIdx.y, cuda.blockIdx.x

    gid = block_i * cuda.gridDim.x + block_j

    prev = active_block[gid]
    current = active_block[gid + 1]
    if current > prev:
        list_data[prev, 0] = block_i
        list_data[prev, 1] = block_j

    prev = active_block_odd[gid]
    current = active_block_odd[gid + 1]
    if current > prev:
        list_data[offset + prev, 0] = block_i
        list_data[offset + prev, 1] = block_j


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
