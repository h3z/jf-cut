import numba
import numpy as np
from numba import cuda

from grabcut.jf_bfs.jf_bfs_cuda.cfg import *

"""
根据bfs得到的block优先级顺序，做全局的relabel（其实是block级别的全局，实际block内部还是无序）
"""


def cuda_global_relabel_jf_init(block_nearest_yx, block_nearest_distance, node_excess, node_label_height):
    img_shape = node_excess.shape
    block_shape = (BLOCK_SIZE, BLOCK_SIZE)
    grid_shape = ((img_shape[0] - 1) // BLOCK_SIZE + 1,
                  (img_shape[1] - 1) // BLOCK_SIZE + 1)

    cuda_global_relabel_jf_init_each_block[grid_shape, block_shape](
        block_nearest_yx, block_nearest_distance, node_excess, node_label_height)


def cuda_global_relabel(node_capacity, node_height, list_data, histogram_data, max_distance):
    done = np.zeros(1, dtype=bool)
    _from, _end, _step = (0, max_distance, 1)
    while not done[0]:
        done[0] = True
        for i in range(_from, _end, _step):
            start = histogram_data[i]
            end = histogram_data[i + 1]
            if start == end:
                continue
            cuda_global_relabel_each_block[end - start, (BLOCK_SIZE, BLOCK_SIZE)](
                start, list_data, node_height, node_capacity, done)

        _from, _end, _step = (_end - _step, _from - _step, -_step)


@cuda.jit
def cuda_global_relabel_jf_init_each_block(block_nearest_yx, block_nearest_distance, node_excess, node_label_height):
    img_shape = node_excess.shape
    node_j, node_i = cuda.grid(2)
    if node_i >= img_shape[0] or node_j >= img_shape[1]:
        return

    block_done = cuda.shared.array(1, dtype=numba.boolean)
    if cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0:
        block_done[0] = False
    cuda.syncthreads()

    if node_excess[node_i, node_j] < 0:
        block_done[0] = True
        node_label_height[node_i, node_j] = 0
    else:
        node_label_height[node_i, node_j] = MAX
    cuda.syncthreads()

    if cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0:
        block_i = cuda.blockIdx.y
        block_j = cuda.blockIdx.x
        if block_done[0]:
            block_nearest_yx[block_i, block_j] = (block_i, block_j)
            block_nearest_distance[block_i, block_j] = 0
        else:
            block_nearest_yx[block_i, block_j] = (MAX, MAX)
            block_nearest_distance[block_i, block_j] = MAX


@cuda.jit
def cuda_global_relabel_each_block(offset, list_data, node_height, node_capacity, done):
    img_shape = node_height.shape
    block_i, block_j = list_data[cuda.blockIdx.x + offset]
    offset_i = block_i * BLOCK_SIZE
    offset_j = block_j * BLOCK_SIZE
    shared_i = cuda.threadIdx.y
    shared_j = cuda.threadIdx.x
    node_i = offset_i + shared_i
    node_j = offset_j + shared_j
    if node_i >= img_shape[0] or node_j >= img_shape[1]:
        return
    block_done = cuda.shared.array(1, dtype=numba.boolean)
    shared_node_height = cuda.shared.array((16 + 2, 16 + 2), dtype=numba.uint32)
    h = node_height[node_i, node_j]
    old_h = h

    shared_node_height[shared_i + 1, shared_j + 1] = h
    if h > 0:
        if shared_i == 0:  # up
            shared_node_height[shared_i, shared_j + 1] = node_height[node_i - 1, node_j] if node_i > 0 else 0
        if shared_j == 0:  # left
            shared_node_height[shared_i + 1, shared_j] = node_height[node_i, node_j - 1] if node_j > 0 else 0
        if node_i == min(offset_i + BLOCK_SIZE, img_shape[0]) - 1:  # down
            if node_i < img_shape[0] - 1:
                shared_node_height[shared_i + 2, shared_j + 1] = node_height[node_i + 1, node_j]
            else:
                shared_node_height[shared_i + 2, shared_j + 1] = 0
        if node_j == min(offset_j + BLOCK_SIZE, img_shape[1]) - 1:  # right
            if node_j < img_shape[1] - 1:
                shared_node_height[shared_i + 1, shared_j + 2] = node_height[node_i, node_j + 1]
            else:
                shared_node_height[shared_i + 1, shared_j + 2] = 0

    if shared_i == 0 and shared_j == 0:
        block_done[0] = False

    cuda.syncthreads()

    global_done = True
    while not block_done[0]:

        cuda.syncthreads()

        block_done[0] = True

        cuda.syncthreads()

        if h > 0:
            c = node_capacity[node_i, node_j]
            t = h
            if c[0] > 0:  # up
                h = min(h, shared_node_height[shared_i, shared_j + 1] + 1)
            if c[1] > 0:  # down
                h = min(h, shared_node_height[shared_i + 2, shared_j + 1] + 1)
            if c[2] > 0:  # left
                h = min(h, shared_node_height[shared_i + 1, shared_j] + 1)
            if c[3] > 0:  # right
                h = min(h, shared_node_height[shared_i + 1, shared_j + 2] + 1)

            if h != t:
                shared_node_height[shared_i + 1, shared_j + 1] = h
                block_done[0] = False
                global_done = False

        cuda.syncthreads()

    if not global_done:
        block_done[0] = False

    cuda.syncthreads()

    if shared_i == 0 and shared_j == 0:
        if not block_done[0]:
            done[0] = False

    if h != old_h:
        node_height[node_i, node_j] = h
