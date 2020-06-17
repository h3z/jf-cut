import numpy as np
import numba
from grabcut.jf_bfs.jf_bfs_seq.cfg import *


@numba.jit
def relabel_push_each_block(node_excess, node_height, node_capacity, block_i, block_j):
    img_shape = node_excess.shape
    offset_i = block_i * BLOCK_SIZE
    offset_j = block_j * BLOCK_SIZE

    shared_node_height = np.zeros((BLOCK_SIZE + 2, BLOCK_SIZE + 2), dtype=np.int64)
    __shared_node_excess = np.zeros((BLOCK_SIZE + 2, BLOCK_SIZE + 2))
    shared_node_capacity = np.zeros((BLOCK_SIZE + 2, BLOCK_SIZE + 2, 4))

    for node_i in range(offset_i, offset_i + BLOCK_SIZE):
        for node_j in range(offset_j, offset_j + BLOCK_SIZE):
            if node_i >= img_shape[0] or node_j >= img_shape[1]:
                break
            shared_i, shared_j = node_i - offset_i, node_j - offset_j
            if node_height[node_i, node_j] < MAX:
                shared_node_height[shared_i + 1, shared_j + 1] = node_height[node_i, node_j]
                __shared_node_excess[shared_i + 1, shared_j + 1] = node_excess[node_i, node_j]
                shared_node_capacity[shared_i + 1, shared_j + 1] = node_capacity[node_i, node_j]

                if node_i == offset_i:  # up
                    __shared_node_excess[shared_i, shared_j + 1] = 0
                    shared_node_height[shared_i, shared_j + 1] = node_height[node_i - 1, node_j] if node_i > 0 else 0
                if node_j == offset_j:  # left
                    __shared_node_excess[shared_i + 1, shared_j] = 0
                    shared_node_height[shared_i + 1, shared_j] = node_height[node_i, node_j - 1] if node_j > 0 else 0
                if node_i == min(offset_i + BLOCK_SIZE, img_shape[0]) - 1:  # down
                    __shared_node_excess[shared_i + 2, shared_j + 1] = 0
                    shared_node_height[shared_i + 2, shared_j + 1] = (
                        node_height[node_i + 1, node_j] if node_i < img_shape[0] - 1 else 0)
                if node_j == min(offset_j + BLOCK_SIZE, img_shape[1]) - 1:  # right
                    __shared_node_excess[shared_i + 1, shared_j + 2] = 0
                    shared_node_height[shared_i + 1, shared_j + 2] = (
                        node_height[node_i, node_j + 1] if node_j < img_shape[1] - 1 else 0)
            else:
                shared_node_height[shared_i + 1, shared_j + 1] = MAX

    block_done = False
    while not block_done:
        block_done = True

        for node_i in range(offset_i, offset_i + BLOCK_SIZE):
            for node_j in range(offset_j, offset_j + BLOCK_SIZE):
                if node_i >= img_shape[0] or node_j >= img_shape[1]:
                    break
                shared_i, shared_j = node_i - offset_i, node_j - offset_j

                parity = (node_i + node_j) & 1
                excess = __shared_node_excess[shared_i + 1, shared_j + 1]
                is_active = excess >= 0 and shared_node_height[shared_i + 1, shared_j + 1] < MAX
                min_height = MAX
                c = shared_node_capacity[shared_i + 1, shared_j + 1]
                if is_active and not parity:
                    if c[0] > 0:  # up
                        min_height = min(min_height, shared_node_height[shared_i, shared_j + 1] + 1)
                    if c[1] > 0:  # down
                        min_height = min(min_height, shared_node_height[shared_i + 2, shared_j + 1] + 1)
                    if c[2] > 0:  # left
                        min_height = min(min_height, shared_node_height[shared_i + 1, shared_j] + 1)
                    if c[3] > 0:  # right
                        min_height = min(min_height, shared_node_height[shared_i + 1, shared_j + 2] + 1)
                    shared_node_height[shared_i + 1, shared_j + 1] = min_height

        for node_i in range(offset_i, offset_i + BLOCK_SIZE):
            for node_j in range(offset_j, offset_j + BLOCK_SIZE):
                if node_i >= img_shape[0] or node_j >= img_shape[1]:
                    break
                shared_i, shared_j = node_i - offset_i, node_j - offset_j

                parity = (node_i + node_j) & 1
                excess = __shared_node_excess[shared_i + 1, shared_j + 1]
                is_active = excess >= 0 and shared_node_height[shared_i + 1, shared_j + 1] < MAX
                min_height = MAX
                c = shared_node_capacity[shared_i + 1, shared_j + 1]

                if is_active and parity:
                    if c[0] > 0:  # up
                        min_height = min(min_height, shared_node_height[shared_i, shared_j + 1] + 1)
                    if c[1] > 0:  # down
                        min_height = min(min_height, shared_node_height[shared_i + 2, shared_j + 1] + 1)
                    if c[2] > 0:  # left
                        min_height = min(min_height, shared_node_height[shared_i + 1, shared_j] + 1)
                    if c[3] > 0:  # right
                        min_height = min(min_height, shared_node_height[shared_i + 1, shared_j + 2] + 1)
                    shared_node_height[shared_i + 1, shared_j + 1] = min_height

        for node_i in range(offset_i, offset_i + BLOCK_SIZE):
            for node_j in range(offset_j, offset_j + BLOCK_SIZE):
                if node_i >= img_shape[0] or node_j >= img_shape[1]:
                    break
                shared_i, shared_j = node_i - offset_i, node_j - offset_j

                parity = (node_i + node_j) & 1
                excess = __shared_node_excess[shared_i + 1, shared_j + 1]
                is_active = excess >= 0 and shared_node_height[shared_i + 1, shared_j + 1] < MAX
                c = shared_node_capacity[shared_i + 1, shared_j + 1]
                min_height = shared_node_height[shared_i + 1, shared_j + 1]

                if is_active and not parity and excess > 0:
                    # up
                    t = min(c[0], excess)
                    if t > 0 and min_height - 1 == shared_node_height[shared_i, shared_j + 1]:
                        shared_node_capacity[shared_i + 1, shared_j + 1, 0] -= t
                        __shared_node_excess[shared_i + 1, shared_j + 1] -= t
                        shared_node_capacity[shared_i, shared_j + 1, 1] += t
                        # TODO 这里要加atomic
                        __shared_node_excess[shared_i, shared_j + 1] += t
                        excess -= t
                        block_done = False
                    # down
                    t = min(c[1], excess)
                    if t > 0 and min_height - 1 == shared_node_height[shared_i + 2, shared_j + 1]:
                        shared_node_capacity[shared_i + 1, shared_j + 1, 1] -= t
                        __shared_node_excess[shared_i + 1, shared_j + 1] -= t
                        shared_node_capacity[shared_i + 2, shared_j + 1, 0] += t
                        # TODO 这里要加atomic
                        __shared_node_excess[shared_i + 2, shared_j + 1] += t
                        excess -= t
                        block_done = False
                    # left
                    t = min(c[2], excess)
                    if t > 0 and min_height - 1 == shared_node_height[shared_i + 1, shared_j]:
                        shared_node_capacity[shared_i + 1, shared_j + 1, 2] -= t
                        __shared_node_excess[shared_i + 1, shared_j + 1] -= t
                        shared_node_capacity[shared_i + 1, shared_j, 3] += t
                        # TODO 这里要加atomic
                        __shared_node_excess[shared_i + 1, shared_j] += t
                        excess -= t
                        block_done = False
                    # right
                    t = min(c[3], excess)
                    if t > 0 and min_height - 1 == shared_node_height[shared_i + 1, shared_j + 2]:
                        shared_node_capacity[shared_i + 1, shared_j + 1, 3] -= t
                        __shared_node_excess[shared_i + 1, shared_j + 1] -= t
                        shared_node_capacity[shared_i + 1, shared_j + 2, 2] += t
                        # TODO 这里要加atomic
                        __shared_node_excess[shared_i + 1, shared_j + 2] += t
                        excess -= t
                        block_done = False

        for node_i in range(offset_i, offset_i + BLOCK_SIZE):
            for node_j in range(offset_j, offset_j + BLOCK_SIZE):
                if node_i >= img_shape[0] or node_j >= img_shape[1]:
                    break
                shared_i, shared_j = node_i - offset_i, node_j - offset_j

                parity = (node_i + node_j) & 1
                excess = __shared_node_excess[shared_i + 1, shared_j + 1]
                is_active = excess >= 0 and shared_node_height[shared_i + 1, shared_j + 1] < MAX
                min_height = shared_node_height[shared_i + 1, shared_j + 1]
                c = shared_node_capacity[shared_i + 1, shared_j + 1]

                if is_active and parity and excess > 0:
                    # up
                    t = min(c[0], excess)
                    if t > 0 and min_height - 1 == shared_node_height[shared_i, shared_j + 1]:
                        shared_node_capacity[shared_i + 1, shared_j + 1, 0] -= t
                        __shared_node_excess[shared_i + 1, shared_j + 1] -= t
                        shared_node_capacity[shared_i, shared_j + 1, 1] += t
                        # TODO 这里要加atomic
                        __shared_node_excess[shared_i, shared_j + 1] += t
                        excess -= t
                        block_done = False
                    # down
                    t = min(c[1], excess)
                    if t > 0 and min_height - 1 == shared_node_height[shared_i + 2, shared_j + 1]:
                        shared_node_capacity[shared_i + 1, shared_j + 1, 1] -= t
                        __shared_node_excess[shared_i + 1, shared_j + 1] -= t
                        shared_node_capacity[shared_i + 2, shared_j + 1, 0] += t
                        # TODO 这里要加atomic
                        __shared_node_excess[shared_i + 2, shared_j + 1] += t
                        excess -= t
                        block_done = False
                    # left
                    t = min(c[2], excess)
                    if t > 0 and min_height - 1 == shared_node_height[shared_i + 1, shared_j]:
                        shared_node_capacity[shared_i + 1, shared_j + 1, 2] -= t
                        __shared_node_excess[shared_i + 1, shared_j + 1] -= t
                        shared_node_capacity[shared_i + 1, shared_j, 3] += t
                        # TODO 这里要加atomic
                        __shared_node_excess[shared_i + 1, shared_j] += t
                        excess -= t
                        block_done = False
                    # right
                    t = min(c[3], excess)
                    if t > 0 and min_height - 1 == shared_node_height[shared_i + 1, shared_j + 2]:
                        shared_node_capacity[shared_i + 1, shared_j + 1, 3] -= t
                        __shared_node_excess[shared_i + 1, shared_j + 1] -= t
                        shared_node_capacity[shared_i + 1, shared_j + 2, 2] += t
                        # TODO 这里要加atomic
                        __shared_node_excess[shared_i + 1, shared_j + 2] += t
                        excess -= t
                        block_done = False

    for node_i in range(offset_i, offset_i + BLOCK_SIZE):
        for node_j in range(offset_j, offset_j + BLOCK_SIZE):
            if node_i >= img_shape[0] or node_j >= img_shape[1]:
                break
            if node_height[node_i, node_j] < MAX:
                shared_i, shared_j = node_i - offset_i, node_j - offset_j
                if shared_i == 0:  # up
                    f = __shared_node_excess[shared_i, shared_j + 1]
                    if f > 0:
                        node_excess[node_i - 1, node_j] += f  # TODO atomic
                        node_capacity[node_i - 1, node_j, 1] += f  # TODO atomic
                if shared_i == BLOCK_SIZE - 1:  # down
                    f = __shared_node_excess[shared_i + 2, shared_j + 1]
                    if f > 0:
                        node_excess[node_i + 1, node_j] += f  # TODO atomic
                        node_capacity[node_i + 1, node_j, 0] += f  # TODO atomic
                if shared_j == BLOCK_SIZE - 1:  # right
                    f = __shared_node_excess[shared_i + 1, shared_j + 2]
                    if f > 0:
                        node_excess[node_i, node_j + 1] += f  # TODO atomic
                        node_capacity[node_i, node_j + 1, 2] += f  # TODO atomic
                if shared_j == 0:  # left
                    f = __shared_node_excess[shared_i + 1, shared_j]
                    if f > 0:
                        node_excess[node_i, node_j - 1] += f  # TODO atomic
                        node_capacity[node_i, node_j - 1, 3] += f  # TODO atomic

                if __shared_node_excess[shared_i + 1, shared_j + 1] != node_excess[node_i, node_j]:
                    node_excess[node_i, node_j] = __shared_node_excess[shared_i + 1, shared_j + 1]
                if shared_node_height[shared_i + 1, shared_j + 1] != node_height[node_i, node_j]:
                    node_height[node_i, node_j] = shared_node_height[shared_i + 1, shared_j + 1]
                if np.any(shared_node_capacity[shared_i + 1, shared_j + 1] != node_capacity[node_i, node_j]):
                    node_capacity[node_i, node_j] = shared_node_capacity[shared_i + 1, shared_j + 1]
            # else:
            #     print('......')


def relabel_push_blocks(node_excess, node_height, node_capacity, list_data, start, end):
    for i in range(start, end):
        block_i, block_j = list_data[i]
        relabel_push_each_block(node_excess, node_height, node_capacity, block_i, block_j)


def relabel_push(list_data, block_offset, node_excess, node_height, node_capacity):
    for i in range(1, block_offset.shape[0]):
        start = block_offset[i - 1]
        end = block_offset[i]
        if start == end:
            continue
        relabel_push_blocks(node_excess, node_height, node_capacity, list_data, start, end)
