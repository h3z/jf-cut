import numba
from numba import cuda

from grabcut.jf_bfs.jf_bfs_cuda.cfg import *


@cuda.jit
def cuda_relabel_push_each_block(offset, list_data, node_excess, node_height, node_capacity):
    img_shape = node_excess.shape
    block_i, block_j = list_data[cuda.blockIdx.x + offset]
    offset_i = block_i * BLOCK_SIZE
    offset_j = block_j * BLOCK_SIZE
    shared_i = cuda.threadIdx.y
    shared_j = cuda.threadIdx.x
    node_i = offset_i + shared_i
    node_j = offset_j + shared_j

    if node_i >= img_shape[0] or node_j >= img_shape[1]:
        return
    parity = (node_i + node_j) & 1
    __shared_node_height = cuda.shared.array((16 + 2, 16 + 2), dtype=numba.uint32)
    __shared_node_excess = cuda.shared.array((16 + 2, 16 + 2), dtype=numba.float32)
    shared_node_capacity = cuda.shared.array((16 + 2, 16 + 2, 4), dtype=numba.float32)
    block_done = cuda.shared.array(1, dtype=numba.boolean)

    h = node_height[node_i, node_j]
    old_h = h
    old_excess = node_excess[node_i, node_j]
    old_c = node_capacity[node_i, node_j]
    if h < MAX:
        __shared_node_height[shared_i + 1, shared_j + 1] = h
        __shared_node_excess[shared_i + 1, shared_j + 1] = old_excess
        shared_node_capacity[shared_i + 1, shared_j + 1, 0] = node_capacity[node_i, node_j, 0]
        shared_node_capacity[shared_i + 1, shared_j + 1, 1] = node_capacity[node_i, node_j, 1]
        shared_node_capacity[shared_i + 1, shared_j + 1, 2] = node_capacity[node_i, node_j, 2]
        shared_node_capacity[shared_i + 1, shared_j + 1, 3] = node_capacity[node_i, node_j, 3]

        if node_i == offset_i:  # up
            __shared_node_excess[shared_i, shared_j + 1] = 0
            __shared_node_height[shared_i, shared_j + 1] = node_height[node_i - 1, node_j] if node_i > 0 else 0
        if node_j == offset_j:  # left
            __shared_node_excess[shared_i + 1, shared_j] = 0
            __shared_node_height[shared_i + 1, shared_j] = node_height[node_i, node_j - 1] if node_j > 0 else 0
        if node_i == min(offset_i + BLOCK_SIZE, img_shape[0]) - 1:  # down
            __shared_node_excess[shared_i + 2, shared_j + 1] = 0
            __shared_node_height[shared_i + 2, shared_j + 1] = (
                node_height[node_i + 1, node_j] if node_i < img_shape[0] - 1 else 0)
        if node_j == min(offset_j + BLOCK_SIZE, img_shape[1]) - 1:  # right
            __shared_node_excess[shared_i + 1, shared_j + 2] = 0
            __shared_node_height[shared_i + 1, shared_j + 2] = (
                node_height[node_i, node_j + 1] if node_j < img_shape[1] - 1 else 0)
    else:
        __shared_node_height[shared_i + 1, shared_j + 1] = MAX

    if shared_i == 0 and shared_j == 0:
        block_done[0] = False

    cuda.syncthreads()

    # count = 0
    while not block_done[0]:
        # count += 1
        excess = __shared_node_excess[shared_i + 1, shared_j + 1]

        is_active = excess >= 0 and __shared_node_height[shared_i + 1, shared_j + 1] < MAX

        # if offset_i == 48 and offset_j == 240 and shared_i == 2 and shared_j == 8:
        #     print('excess', __shared_node_excess[shared_i + 1, shared_j + 1])
        #     print('up excess, ', __shared_node_excess[shared_i, shared_j + 1])
        #     print('down excess, ', __shared_node_excess[shared_i + 2, shared_j + 1])
        #     print('left excess, ', __shared_node_excess[shared_i + 1, shared_j])
        #     print('right excess, ', __shared_node_excess[shared_i + 1, shared_j + 2])
        #     print('height, ', __shared_node_height[shared_i + 1, shared_j + 1])
        #     print('up height, ', __shared_node_height[shared_i, shared_j + 1])
        #     print('down height, ', __shared_node_height[shared_i + 2, shared_j + 1])
        #     print('left height, ', __shared_node_height[shared_i + 1, shared_j])
        #     print('right height, ', __shared_node_height[shared_i + 1, shared_j + 2])
        #     print('capacity up, ', shared_node_capacity[shared_i + 1, shared_j + 1, 0])
        #     print('capacity down, ', shared_node_capacity[shared_i + 1, shared_j + 1, 1])
        #     print('capacity left, ', shared_node_capacity[shared_i + 1, shared_j + 1, 2])
        #     print('capacity right, ', shared_node_capacity[shared_i + 1, shared_j + 1, 3])
        # cuda.syncthreads()

        # relabel1
        if is_active and not parity:
            c = shared_node_capacity[shared_i + 1, shared_j + 1]
            h = MAX
            if c[0] > 0:  # up
                h = min(h, __shared_node_height[shared_i, shared_j + 1] + 1)
            if c[1] > 0:  # down
                h = min(h, __shared_node_height[shared_i + 2, shared_j + 1] + 1)
            if c[2] > 0:  # left
                h = min(h, __shared_node_height[shared_i + 1, shared_j] + 1)
            if c[3] > 0:  # right
                h = min(h, __shared_node_height[shared_i + 1, shared_j + 2] + 1)

            __shared_node_height[shared_i + 1, shared_j + 1] = h
        cuda.syncthreads()
        # if offset_i == 48 and offset_j == 240 and shared_i == 2 and shared_j == 8:
        #     print('after excess', __shared_node_excess[shared_i + 1, shared_j + 1])
        #     print('after up excess, ', __shared_node_excess[shared_i, shared_j + 1])
        #     print('after down excess, ', __shared_node_excess[shared_i + 2, shared_j + 1])
        #     print('after left excess, ', __shared_node_excess[shared_i + 1, shared_j])
        #     print('after right excess, ', __shared_node_excess[shared_i + 1, shared_j + 2])
        #     print('after height, ', __shared_node_height[shared_i + 1, shared_j + 1])
        #     print('after up height, ', __shared_node_height[shared_i, shared_j + 1])
        #     print('after down height, ', __shared_node_height[shared_i + 2, shared_j + 1])
        #     print('after left height, ', __shared_node_height[shared_i + 1, shared_j])
        #     print('after right height, ', __shared_node_height[shared_i + 1, shared_j + 2])
        #     print('after capacity up, ', shared_node_capacity[shared_i + 1, shared_j + 1, 0])
        #     print('after capacity down, ', shared_node_capacity[shared_i + 1, shared_j + 1, 1])
        #     print('after capacity left, ', shared_node_capacity[shared_i + 1, shared_j + 1, 2])
        #     print('after capacity right, ', shared_node_capacity[shared_i + 1, shared_j + 1, 3])
        # cuda.syncthreads()

        # relabel2
        if is_active and parity:
            c = shared_node_capacity[shared_i + 1, shared_j + 1]
            h = MAX
            if c[0] > 0:  # up
                h = min(h, __shared_node_height[shared_i, shared_j + 1] + 1)
            if c[1] > 0:  # down
                h = min(h, __shared_node_height[shared_i + 2, shared_j + 1] + 1)
            if c[2] > 0:  # left
                h = min(h, __shared_node_height[shared_i + 1, shared_j] + 1)
            if c[3] > 0:  # right
                h = min(h, __shared_node_height[shared_i + 1, shared_j + 2] + 1)

            __shared_node_height[shared_i + 1, shared_j + 1] = h

        block_done[0] = True
        cuda.syncthreads()

        # push1
        is_active = is_active and h < MAX
        if is_active and not parity and excess > 0:

            # up
            c = shared_node_capacity[shared_i + 1, shared_j + 1]
            t = min(c[0], excess)
            if t > 0 and h - 1 == __shared_node_height[shared_i, shared_j + 1]:
                shared_node_capacity[shared_i + 1, shared_j + 1, 0] -= t
                shared_node_capacity[shared_i, shared_j + 1, 1] += t
                excess -= t
                __shared_node_excess[shared_i + 1, shared_j + 1] -= t
                cuda.atomic.add(__shared_node_excess, (shared_i, shared_j + 1), t)
                block_done[0] = False
            # down
            t = min(c[1], excess)
            if t > 0 and h - 1 == __shared_node_height[shared_i + 2, shared_j + 1]:
                shared_node_capacity[shared_i + 1, shared_j + 1, 1] -= t
                shared_node_capacity[shared_i + 2, shared_j + 1, 0] += t
                excess -= t
                __shared_node_excess[shared_i + 1, shared_j + 1] -= t
                cuda.atomic.add(__shared_node_excess, (shared_i + 2, shared_j + 1), t)
                block_done[0] = False
            # left
            t = min(c[2], excess)
            if t > 0 and h - 1 == __shared_node_height[shared_i + 1, shared_j]:
                shared_node_capacity[shared_i + 1, shared_j + 1, 2] -= t
                shared_node_capacity[shared_i + 1, shared_j, 3] += t
                excess -= t
                __shared_node_excess[shared_i + 1, shared_j + 1] -= t
                cuda.atomic.add(__shared_node_excess, (shared_i + 1, shared_j), t)
                block_done[0] = False
            # right
            t = min(c[3], excess)
            if t > 0 and h - 1 == __shared_node_height[shared_i + 1, shared_j + 2]:
                shared_node_capacity[shared_i + 1, shared_j + 1, 3] -= t
                shared_node_capacity[shared_i + 1, shared_j + 2, 2] += t
                excess -= t
                __shared_node_excess[shared_i + 1, shared_j + 1] -= t
                cuda.atomic.add(__shared_node_excess, (shared_i + 1, shared_j + 2), t)
                block_done[0] = False

        cuda.syncthreads()

        # if offset_i == 48 and offset_j == 240 and shared_i == 3 and shared_j == 7:
        #     print('excess', __shared_node_excess[shared_i + 1, shared_j + 1])
        #     print('up excess, ', __shared_node_excess[shared_i, shared_j + 1])
        #     print('down excess, ', __shared_node_excess[shared_i + 2, shared_j + 1])
        #     print('left excess, ', __shared_node_excess[shared_i + 1, shared_j])
        #     print('right excess, ', __shared_node_excess[shared_i + 1, shared_j + 2])
        #     print('height, ', __shared_node_height[shared_i + 1, shared_j + 1])
        #     print('up height, ', __shared_node_height[shared_i, shared_j + 1])
        #     print('down height, ', __shared_node_height[shared_i + 2, shared_j + 1])
        #     print('left height, ', __shared_node_height[shared_i + 1, shared_j])
        #     print('right height, ', __shared_node_height[shared_i + 1, shared_j + 2])
        #     print('capacity from up, ', shared_node_capacity[shared_i, shared_j + 1, 1])
        #     print('capacity from down, ', shared_node_capacity[shared_i + 2, shared_j + 1, 0])
        #     print('capacity from left, ', shared_node_capacity[shared_i + 1, shared_j, 3])
        #     print('capacity from right, ', shared_node_capacity[shared_i + 1, shared_j + 2, 2])
        # cuda.syncthreads()

        excess = __shared_node_excess[shared_i + 1, shared_j + 1]
        # push2
        if is_active and parity and excess > 0:
            # up
            c = shared_node_capacity[shared_i + 1, shared_j + 1]
            t = min(c[0], excess)
            if t > 0 and h - 1 == __shared_node_height[shared_i, shared_j + 1]:
                shared_node_capacity[shared_i + 1, shared_j + 1, 0] -= t
                shared_node_capacity[shared_i, shared_j + 1, 1] += t
                excess -= t
                __shared_node_excess[shared_i + 1, shared_j + 1] -= t
                cuda.atomic.add(__shared_node_excess, (shared_i, shared_j + 1), t)
                block_done[0] = False
            # down
            t = min(c[1], excess)
            if t > 0 and h - 1 == __shared_node_height[shared_i + 2, shared_j + 1]:
                shared_node_capacity[shared_i + 1, shared_j + 1, 1] -= t
                shared_node_capacity[shared_i + 2, shared_j + 1, 0] += t
                excess -= t
                __shared_node_excess[shared_i + 1, shared_j + 1] -= t
                cuda.atomic.add(__shared_node_excess, (shared_i + 2, shared_j + 1), t)
                block_done[0] = False
            # left
            t = min(c[2], excess)
            if t > 0 and h - 1 == __shared_node_height[shared_i + 1, shared_j]:
                shared_node_capacity[shared_i + 1, shared_j + 1, 2] -= t
                shared_node_capacity[shared_i + 1, shared_j, 3] += t
                excess -= t
                __shared_node_excess[shared_i + 1, shared_j + 1] -= t
                cuda.atomic.add(__shared_node_excess, (shared_i + 1, shared_j), t)
                block_done[0] = False
            # right
            t = min(c[3], excess)
            if t > 0 and h - 1 == __shared_node_height[shared_i + 1, shared_j + 2]:
                shared_node_capacity[shared_i + 1, shared_j + 1, 3] -= t
                shared_node_capacity[shared_i + 1, shared_j + 2, 2] += t
                excess -= t
                __shared_node_excess[shared_i + 1, shared_j + 1] -= t
                cuda.atomic.add(__shared_node_excess, (shared_i + 1, shared_j + 2), t)
                block_done[0] = False

        cuda.syncthreads()

        # if offset_i == 48 and offset_j == 240 and shared_i == 3 and shared_j == 7:
        #     print('after excess', __shared_node_excess[shared_i + 1, shared_j + 1])
        #     print('after up excess, ', __shared_node_excess[shared_i, shared_j + 1])
        #     print('after down excess, ', __shared_node_excess[shared_i + 2, shared_j + 1])
        #     print('after left excess, ', __shared_node_excess[shared_i + 1, shared_j])
        #     print('after right excess, ', __shared_node_excess[shared_i + 1, shared_j + 2])
        #     print('after height, ', __shared_node_height[shared_i + 1, shared_j + 1])
        #     print('after up height, ', __shared_node_height[shared_i, shared_j + 1])
        #     print('after down height, ', __shared_node_height[shared_i + 2, shared_j + 1])
        #     print('after left height, ', __shared_node_height[shared_i + 1, shared_j])
        #     print('after right height, ', __shared_node_height[shared_i + 1, shared_j + 2])
        #     print('after capacity from up, ', shared_node_capacity[shared_i, shared_j + 1, 1])
        #     print('after capacity from down, ', shared_node_capacity[shared_i + 2, shared_j + 1, 0])
        #     print('after capacity from left, ', shared_node_capacity[shared_i + 1, shared_j, 3])
        #     print('after capacity from right, ', shared_node_capacity[shared_i + 1, shared_j + 2, 2])
        # cuda.syncthreads()

    # update global memory
    if old_h < MAX:
        if shared_i == 0:  # up
            f = __shared_node_excess[shared_i, shared_j + 1]
            if f > 0:
                cuda.atomic.add(node_excess, (node_i - 1, node_j), f)
                node_capacity[node_i - 1, node_j, 1] += f
        if shared_i == BLOCK_SIZE - 1:  # down
            f = __shared_node_excess[shared_i + 2, shared_j + 1]
            if f > 0:
                cuda.atomic.add(node_excess, (node_i + 1, node_j), f)
                node_capacity[node_i + 1, node_j, 0] += f
        if shared_j == BLOCK_SIZE - 1:  # right
            f = __shared_node_excess[shared_i + 1, shared_j + 2]
            if f > 0:
                cuda.atomic.add(node_excess, (node_i, node_j + 1), f)
                node_capacity[node_i, node_j + 1, 2] += f
        if shared_j == 0:  # left
            f = __shared_node_excess[shared_i + 1, shared_j]
            if f > 0:
                cuda.atomic.add(node_excess, (node_i, node_j - 1), f)
                node_capacity[node_i, node_j - 1, 3] += f

        excess = __shared_node_excess[shared_i + 1, shared_j + 1]
        c = shared_node_capacity[shared_i + 1, shared_j + 1]
        height = __shared_node_height[shared_i + 1, shared_j + 1]
        if excess != old_excess:
            node_excess[node_i, node_j] = excess
        if height != old_h:
            node_height[node_i, node_j] = height
        if c[0] != old_c[0]:
            node_capacity[node_i, node_j, 0] = c[0]
        if c[1] != old_c[1]:
            node_capacity[node_i, node_j, 1] = c[1]
        if c[2] != old_c[2]:
            node_capacity[node_i, node_j, 2] = c[2]
        if c[3] != old_c[3]:
            node_capacity[node_i, node_j, 3] = c[3]


@cuda.jit
def cuda_relabel_push_each_block2(offset, list_data, node_excess, node_height, node_capacity):
    img_shape = node_excess.shape
    block_i, block_j = list_data[cuda.blockIdx.x + offset]
    offset_i = block_i * BLOCK_SIZE
    offset_j = block_j * BLOCK_SIZE

    shared_node_height = cuda.local.array((16 + 2, 16 + 2), dtype=numba.int64)
    __shared_node_excess = cuda.local.array((16 + 2, 16 + 2), dtype=numba.float32)
    shared_node_capacity = cuda.local.array((16 + 2, 16 + 2, 4), dtype=numba.float32)

    for node_i in range(offset_i, offset_i + BLOCK_SIZE):
        for node_j in range(offset_j, offset_j + BLOCK_SIZE):
            if node_i >= img_shape[0] or node_j >= img_shape[1]:
                break
            shared_i, shared_j = node_i - offset_i, node_j - offset_j
            if node_height[node_i, node_j] < MAX:
                shared_node_height[shared_i + 1, shared_j + 1] = node_height[node_i, node_j]
                __shared_node_excess[shared_i + 1, shared_j + 1] = node_excess[node_i, node_j]
                shared_node_capacity[shared_i + 1, shared_j + 1, 0] = node_capacity[node_i, node_j, 0]
                shared_node_capacity[shared_i + 1, shared_j + 1, 1] = node_capacity[node_i, node_j, 1]
                shared_node_capacity[shared_i + 1, shared_j + 1, 2] = node_capacity[node_i, node_j, 2]
                shared_node_capacity[shared_i + 1, shared_j + 1, 3] = node_capacity[node_i, node_j, 3]

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
    # count = 0
    while not block_done:
        # count += 1
        block_done = True

        # label
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

        # for node_i in range(offset_i, offset_i + BLOCK_SIZE):
        #     for node_j in range(offset_j, offset_j + BLOCK_SIZE):
        #         shared_i, shared_j = node_i - offset_i, node_j - offset_j
        #
        #         if offset_i == 48 and offset_j == 240 and shared_i == 2 and shared_j == 7:
        #             print('excess', __shared_node_excess[shared_i + 1, shared_j + 1])
        #             print('up excess, ', __shared_node_excess[shared_i, shared_j + 1])
        #             print('down excess, ', __shared_node_excess[shared_i + 2, shared_j + 1])
        #             print('left excess, ', __shared_node_excess[shared_i + 1, shared_j])
        #             print('right excess, ', __shared_node_excess[shared_i + 1, shared_j + 2])
        #             print('height, ', shared_node_height[shared_i + 1, shared_j + 1])
        #             print('up height, ', shared_node_height[shared_i, shared_j + 1])
        #             print('down height, ', shared_node_height[shared_i + 2, shared_j + 1])
        #             print('left height, ', shared_node_height[shared_i + 1, shared_j])
        #             print('right height, ', shared_node_height[shared_i + 1, shared_j + 2])
        #             print('capacity up, ', shared_node_capacity[shared_i + 1, shared_j + 1, 0])
        #             print('capacity down, ', shared_node_capacity[shared_i + 1, shared_j + 1, 1])
        #             print('capacity left, ', shared_node_capacity[shared_i + 1, shared_j + 1, 2])
        #             print('capacity right, ', shared_node_capacity[shared_i + 1, shared_j + 1, 3])

        # label
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

        # push
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

        # for node_i in range(offset_i, offset_i + BLOCK_SIZE):
        #     for node_j in range(offset_j, offset_j + BLOCK_SIZE):
        #         shared_i, shared_j = node_i - offset_i, node_j - offset_j
        #
        #         if offset_i == 48 and offset_j == 240 and shared_i == 3 and shared_j == 7:
        #             print('excess', __shared_node_excess[shared_i + 1, shared_j + 1])
        #             print('up excess, ', __shared_node_excess[shared_i, shared_j + 1])
        #             print('down excess, ', __shared_node_excess[shared_i + 2, shared_j + 1])
        #             print('left excess, ', __shared_node_excess[shared_i + 1, shared_j])
        #             print('right excess, ', __shared_node_excess[shared_i + 1, shared_j + 2])
        #             print('height, ', shared_node_height[shared_i + 1, shared_j + 1])
        #             print('up height, ', shared_node_height[shared_i, shared_j + 1])
        #             print('down height, ', shared_node_height[shared_i + 2, shared_j + 1])
        #             print('left height, ', shared_node_height[shared_i + 1, shared_j])
        #             print('right height, ', shared_node_height[shared_i + 1, shared_j + 2])
        #             print('capacity from up, ', shared_node_capacity[shared_i, shared_j + 1, 1])
        #             print('capacity from down, ', shared_node_capacity[shared_i + 2, shared_j + 1, 0])
        #             print('capacity from left, ', shared_node_capacity[shared_i + 1, shared_j, 3])
        #             print('capacity from right, ', shared_node_capacity[shared_i + 1, shared_j + 2, 2])

        # push
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
        # for node_i in range(offset_i, offset_i + BLOCK_SIZE):
        #     for node_j in range(offset_j, offset_j + BLOCK_SIZE):
        #         shared_i, shared_j = node_i - offset_i, node_j - offset_j
        #
        #         if offset_i == 48 and offset_j == 240 and shared_i == 3 and shared_j == 7:
        #             print('after excess', __shared_node_excess[shared_i + 1, shared_j + 1])
        #             print('after up excess, ', __shared_node_excess[shared_i, shared_j + 1])
        #             print('after down excess, ', __shared_node_excess[shared_i + 2, shared_j + 1])
        #             print('after left excess, ', __shared_node_excess[shared_i + 1, shared_j])
        #             print('after right excess, ', __shared_node_excess[shared_i + 1, shared_j + 2])
        #             print('after height, ', shared_node_height[shared_i + 1, shared_j + 1])
        #             print('after up height, ', shared_node_height[shared_i, shared_j + 1])
        #             print('after down height, ', shared_node_height[shared_i + 2, shared_j + 1])
        #             print('after left height, ', shared_node_height[shared_i + 1, shared_j])
        #             print('after right height, ', shared_node_height[shared_i + 1, shared_j + 2])
        #             print('after capacity from up, ', shared_node_capacity[shared_i, shared_j + 1, 1])
        #             print('after capacity from down, ', shared_node_capacity[shared_i + 2, shared_j + 1, 0])
        #             print('after capacity from left, ', shared_node_capacity[shared_i + 1, shared_j, 3])
        #             print('after capacity from right, ', shared_node_capacity[shared_i + 1, shared_j + 2, 2])

    # update global cache
    for node_i in range(offset_i, offset_i + BLOCK_SIZE):
        for node_j in range(offset_j, offset_j + BLOCK_SIZE):
            if node_i >= img_shape[0] or node_j >= img_shape[1]:
                break
            if node_height[node_i, node_j] < MAX:
                shared_i, shared_j = node_i - offset_i, node_j - offset_j
                if shared_i == 0:  # up
                    f = __shared_node_excess[shared_i, shared_j + 1]
                    if f > 0:
                        node_excess[node_i - 1, node_j] += f
                        node_capacity[node_i - 1, node_j, 1] += f
                if shared_i == BLOCK_SIZE - 1:  # down
                    f = __shared_node_excess[shared_i + 2, shared_j + 1]
                    if f > 0:
                        node_excess[node_i + 1, node_j] += f
                        node_capacity[node_i + 1, node_j, 0] += f
                if shared_j == BLOCK_SIZE - 1:  # right
                    f = __shared_node_excess[shared_i + 1, shared_j + 2]
                    if f > 0:
                        node_excess[node_i, node_j + 1] += f
                        node_capacity[node_i, node_j + 1, 2] += f
                if shared_j == 0:  # left
                    f = __shared_node_excess[shared_i + 1, shared_j]
                    if f > 0:
                        node_excess[node_i, node_j - 1] += f
                        node_capacity[node_i, node_j - 1, 3] += f

                if __shared_node_excess[shared_i + 1, shared_j + 1] != node_excess[node_i, node_j]:
                    node_excess[node_i, node_j] = __shared_node_excess[shared_i + 1, shared_j + 1]
                if shared_node_height[shared_i + 1, shared_j + 1] != node_height[node_i, node_j]:
                    node_height[node_i, node_j] = shared_node_height[shared_i + 1, shared_j + 1]

                c = shared_node_capacity[shared_i + 1, shared_j + 1]
                old_c = node_capacity[node_i, node_j]
                if c[0] != old_c[0]:
                    node_capacity[node_i, node_j, 0] = c[0]
                if c[1] != old_c[1]:
                    node_capacity[node_i, node_j, 1] = c[1]
                if c[2] != old_c[2]:
                    node_capacity[node_i, node_j, 2] = c[2]
                if c[3] != old_c[3]:
                    node_capacity[node_i, node_j, 3] = c[3]


def cuda_relabel_push(list_data, block_offset, node_excess, node_height, node_capacity):
    for i in range(1, block_offset.shape[0]):
        start = block_offset[i - 1]
        end = block_offset[i]
        if start == end:
            continue

        cuda_relabel_push_each_block[end - start, (BLOCK_SIZE, BLOCK_SIZE)](
            start, list_data, node_excess, node_height, node_capacity)
        # cuda_relabel_push_each_block2[end - start, 1](
        #     start, list_data, node_excess, node_height, node_capacity)
        # for j in range(start, end):
        #     cuda_relabel_push_each_block[1, (BLOCK_SIZE, BLOCK_SIZE)](
        #         j, list_data, node_excess, node_height, node_capacity)
