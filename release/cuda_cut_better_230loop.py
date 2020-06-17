import math

import numpy as np
from numba import int32

from grabcut.old.sb_cuda_cut_.cuda_push import *

threads_x = 32
threads_y = 8

INACTIVE = 2
ACTIVE = 1
PASSIVE = 0


def init_l_edge(around_capacity):
    weight_left = around_capacity['left']
    weight_up = around_capacity['up']

    h, w = weight_left.shape

    weight_right = np.zeros((h, w))
    weight_right[:h, :w - 1] = weight_left[:, 1:]

    weight_down = np.zeros((h, w))
    weight_down[:h - 1, :w] = weight_up[1:, :]

    return weight_left, weight_up, weight_right, weight_down


@cuda.jit
def init_st_edge(d_bkg_similar, d_frg_similar, d_weight_frg, d_weight_bkg, h, w):
    x, y = cuda.grid(2)
    if 0 <= x < w and 0 <= y < h:
        diff = d_frg_similar[y, x] - d_bkg_similar[y, x]
        if diff > 0:
            d_weight_frg[y, x] = diff
        else:
            d_weight_bkg[y, x] = -diff


@cuda.jit
def push(d_weight_left, d_weight_up, d_weight_right, d_weight_down,
         d_pull_up, d_pull_right, d_pull_down, d_pull_left,
         d_weight_frg, d_weight_bkg, d_height, w, h, d_relabel_mask):
    x, y = cuda.grid(2)
    idx, idy = cuda.threadIdx.x, cuda.threadIdx.y
    if x >= w or y >= h:
        return
    threads_y = cuda.blockDim.y
    threads_x = cuda.blockDim.x
    s_height = cuda.shared.array((8 + 2, 32 + 2), dtype=int32)
    s_height[1 + idy, 1 + idx] = d_height[y, x]

    if idx == threads_x - 1 and x < w - 1:
        s_height[idy + 1, -1] = d_height[y, x + 1]
    if idx == 0 and x > 0:
        s_height[idy + 1, 0] = d_height[y, x - 1]
    if idy == threads_y - 1 and y < h - 1:
        s_height[-1, idx + 1] = d_height[y + 1, x]
    if idy == 0 and y > 0:
        s_height[0, idx + 1] = d_height[y - 1, x]

    cuda.syncthreads()

    flow = d_weight_frg[y, x]
    if 0 < x < w - 1 and 0 < y < h - 1 and d_relabel_mask[y, x] == ACTIVE:
        # -> t
        weight = d_weight_bkg[y, x]
        if weight > 0 and flow > 0 and s_height[idy + 1, idx + 1] == 1:
            min_push = flow if flow < weight else weight
            d_weight_bkg[y, x] -= min_push
            flow -= min_push

        # -> left
        weight = d_weight_left[y, x]
        if weight > 0 and flow > 0 and s_height[idy + 1, idx + 1] == s_height[idy + 1, idx] + 1:
            min_push = flow if flow < weight else weight
            d_weight_left[y, x] -= min_push
            flow -= min_push
            d_pull_left[y, x - 1] = min_push
        else:
            d_pull_left[y, x - 1] = -1

        # -> right
        weight = d_weight_right[y, x]
        if weight > 0 and flow > 0 and s_height[idy + 1, idx + 1] == s_height[idy + 1, idx + 2] + 1:
            min_push = flow if flow < weight else weight
            d_weight_right[y, x] -= min_push
            flow -= min_push
            d_pull_right[y, x + 1] = min_push
        else:
            d_pull_right[y, x + 1] = -1

        # -> up
        weight = d_weight_up[y, x]
        if weight > 0 and flow > 0 and s_height[idy + 1, idx + 1] == s_height[idy, idx + 1] + 1:
            min_push = flow if flow < weight else weight
            d_weight_up[y, x] -= min_push
            flow -= min_push
            d_pull_up[y - 1, x] = min_push
        else:
            d_pull_up[y - 1, x] = -1

        # -> down
        weight = d_weight_down[y, x]
        if weight > 0 and flow > 0 and s_height[idy + 1, idx + 1] == s_height[idy + 2, idx + 1] + 1:
            min_push = flow if flow < weight else weight
            d_weight_down[y, x] -= min_push
            flow -= min_push
            d_pull_down[y + 1, x] = min_push
        else:
            d_pull_down[y + 1, x] = -1
    d_weight_frg[y, x] = flow


@cuda.jit
def local_relabel(d_weight_left, d_weight_up, d_weight_right, d_weight_down,
                  d_pull_up, d_pull_right, d_pull_down, d_pull_left,
                  d_weight_frg, d_weight_bkg, d_height, d_relabel_mask, w, h):
    x, y = cuda.grid(2)
    idx, idy = cuda.threadIdx.x, cuda.threadIdx.y
    if x >= w or y >= h:
        return

    s_height = cuda.shared.array((8 + 2, 32 + 2), dtype=int32)
    s_height[1 + idy, 1 + idx] = d_height[y, x]

    if idx == threads_x - 1 and x < w - 1:
        s_height[idy + 1, -1] = d_height[y, x + 1]
    if idx == 0 and x > 0:
        s_height[idy + 1, 0] = d_height[y, x - 1]
    if idy == threads_y - 1 and y < h - 1:
        s_height[-1, idx + 1] = d_height[y + 1, x]
    if idy == 0 and y > 0:
        s_height[0, idx + 1] = d_height[y - 1, x]

    cuda.syncthreads()

    if 0 < x < w - 1 and 0 < y < h - 1:
        if d_weight_bkg[y, x] > 0:
            d_height[y, x] = 1
        else:
            min_h = w * h
            if d_weight_left[y, x] > 0 and min_h > s_height[idy + 1, idx]:
                min_h = s_height[idy + 1, idx]
            if d_weight_right[y, x] > 0 and min_h > s_height[idy + 1, idx + 2]:
                min_h = s_height[idy + 1, idx + 2]
            if d_weight_up[y, x] > 0 and min_h > s_height[idy, idx + 1]:
                min_h = s_height[idy, idx + 1]
            if d_weight_down[y, x] > 0 and min_h > s_height[idy + 2, idx + 1]:
                min_h = s_height[idy + 2, idx + 1]
            d_height[y, x] = min_h + 1


@cuda.jit
def pull(d_weight_left, d_weight_up, d_weight_right, d_weight_down,
         d_pull_up, d_pull_right, d_pull_down, d_pull_left,
         d_weight_frg, d_weight_bkg, d_height, d_relabel_mask, w, h):
    x, y = cuda.grid(2)
    if x >= w or y >= h:
        return

    flow = d_weight_frg[y, x]
    if 0 < x < w and 0 < y < h:
        flow_left = d_pull_right[y, x]
        flow_right = d_pull_left[y, x]
        flow_up = d_pull_down[y, x]
        flow_down = d_pull_up[y, x]

        d_pull_right[y, x] = 0
        d_pull_left[y, x] = 0
        d_pull_down[y, x] = 0
        d_pull_up[y, x] = 0

        if flow_left > 0:
            d_weight_left[y, x] += flow_left
            flow += flow_left
        if flow_right > 0:
            d_weight_right[y, x] += flow_right
            flow += flow_right
        if flow_down > 0:
            d_weight_down[y, x] += flow_down
            flow += flow_down
        if flow_up > 0:
            d_weight_up[y, x] += flow_up
            flow += flow_up

        d_weight_frg[y, x] = flow

        if flow <= 0 or (d_weight_left[y, x] == 0 and d_weight_right[y, x] == 0 and
                         d_weight_up[y, x] == 0 and d_weight_down[y, x] == 0 and d_weight_bkg[y, x] == 0):
            # if (d_weight_left[y, x] == 0 and d_weight_right[y, x] == 0 and
            #         d_weight_up[y, x] == 0 and d_weight_down[y, x] == 0 and d_weight_bkg[y, x] == 0):
            d_relabel_mask[y, x] = INACTIVE
        elif flow > 0 and (flow_left == -1 and flow_right == -1 and
                           flow_up == -1 and flow_down == -1 and d_height[y, x] != 1):
            d_relabel_mask[y, x] = 0
        else:
            d_relabel_mask[y, x] = 1


@cuda.jit
def global_relabel(d_weight_left, d_weight_up, d_weight_right, d_weight_down,
                   d_pull_up, d_pull_right, d_pull_down, d_pull_left,
                   d_weight_frg, d_weight_bkg, d_height, d_relabel_mask, w, h, k, d_assigned):
    x, y = cuda.grid(2)
    idx, idy = cuda.threadIdx.x, cuda.threadIdx.y
    if x >= w or y >= h:
        return

    s_height = cuda.shared.array((8 + 2, 32 + 2), dtype=int32)
    s_height[1 + idy, 1 + idx] = d_height[y, x]

    if idx == threads_x - 1 and x < w - 1:
        s_height[idy + 1, -1] = d_height[y, x + 1]
    if idx == 0 and x > 0:
        s_height[idy + 1, 0] = d_height[y, x - 1]
    if idy == threads_y - 1 and y < h - 1:
        s_height[-1, idx + 1] = d_height[y + 1, x]
    if idy == 0 and y > 0:
        s_height[0, idx + 1] = d_height[y - 1, x]

    cuda.syncthreads()

    if 0 < x < w - 1 and 0 < y < h - 1:
        if d_weight_bkg[y, x] > 0:
            d_height[y, x] = 1
            d_assigned[y, x] = True
        else:
            if ((d_weight_left[y, x] > 0 and d_assigned[y, x - 1] and s_height[idy + 1, idx] == k) or
                    (d_weight_right[y, x] > 0 and d_assigned[y, x + 1] and s_height[idy + 1, idx + 2] == k) or
                    (d_weight_up[y, x] > 0 and d_assigned[y - 1, x] and s_height[idy, idx + 1] == k) or
                    (d_weight_down[y, x] > 0 and d_assigned[y + 1, x] and s_height[idy + 2, idx + 1] == k)):
                d_weight_left[y, x] = k + 1
                d_assigned[y, x] = True


@cuda.jit
def from_frg_bfs_init(d_weight_left, d_weight_up, d_weight_right, d_weight_down,
                      d_pull_up, d_pull_right, d_pull_down, d_pull_left,
                      d_weight_frg, d_weight_bkg, d_height, d_relabel_mask, w, h, k, d_assigned, d_pixel_mask):
    x, y = cuda.grid(2)
    if x >= w or y >= h:
        return

    if d_pixel_mask[y, x]:
        if d_weight_frg[y, x] > 0:
            d_height[y, x] = 1
            d_pixel_mask[y, x] = False
        elif d_weight_bkg[y, x] > 0:
            d_height[y, x] = -1
            d_pixel_mask[y, x] = False


@cuda.jit
def from_frg_bfs(d_weight_left, d_weight_up, d_weight_right, d_weight_down,
                 d_pull_up, d_pull_right, d_pull_down, d_pull_left,
                 d_weight_frg, d_weight_bkg, d_height, d_relabel_mask, w, h, k, d_assigned, d_pixel_mask, count, over):
    x, y = cuda.grid(2)
    if x >= w or y >= h:
        return

    if not d_pixel_mask[y, x]:
        return
    if 0 < x < w - 1 and 0 < y < h - 1:
        if ((d_height[y, x - 1] == count and d_weight_right[y, x - 1] > 0) or
                (d_height[y, x + 1] == count and d_weight_left[y, x + 1] > 0) or
                (d_height[y + 1, x] == count and d_weight_up[y + 1, x] > 0) or
                (d_height[y - 1, x] == count and d_weight_down[y - 1, x] > 0)):
            d_height[y, x] = count + 1
            d_pixel_mask[y, x] = False
            over[0] = True


@cuda.jit
def kernel_push_stochastic2(d_push_reser, s_push_reser, d_stochastic):
    x, y = cuda.grid(2)

    stochastic = (s_push_reser[y, x] - d_push_reser[y, x])
    if stochastic != 0:
        d_stochastic[cuda.blockIdx.y, cuda.blockIdx.x] = 1


@cuda.jit
def kernel_push_stochastic1(g_push_reser, s_push_reser, g_count_blocks, g_finish):
    x, y = cuda.grid(2)
    s_push_reser[y, x] = g_push_reser[y, x]

    if y == 0 and x == 0:
        if g_count_blocks[0] < 10:
            g_finish[0] = True


def max_flow(img, around_capacity, bkg_similar, frg_similar, m=6, k=2, test=False):
    input_shape = img.shape[:2]
    h, w = input_shape
    block_shape = (threads_x, threads_y)
    grid_shape = (math.ceil(w / threads_x),
                  math.ceil(h / threads_y))

    weight_left, weight_up, weight_right, weight_down = init_l_edge(around_capacity)
    if test:
        weight_right = around_capacity['right']
        weight_down = around_capacity['down']

    d_weight_left = weight_left
    d_weight_up = weight_up
    d_weight_right = weight_right
    d_weight_down = weight_down

    d_pull_up = np.zeros(input_shape)
    d_pull_right = np.zeros(input_shape)
    d_pull_down = np.zeros(input_shape)
    d_pull_left = np.zeros(input_shape)

    relabel_mask = np.ones(input_shape)
    d_relabel_mask = relabel_mask
    assigned = np.zeros(input_shape, dtype=bool)
    d_assigned = assigned

    # d_height = np.ones(input_shape)
    d_height = np.ones(input_shape)
    d_weight_frg = np.zeros(input_shape)
    d_weight_bkg = np.zeros(input_shape)
    d_bkg_similar = bkg_similar
    d_frg_similar = frg_similar

    init_st_edge[grid_shape, block_shape](d_bkg_similar, d_frg_similar, d_weight_frg, d_weight_bkg, h, w)

    # while np.sum(d_relabel_mask == 2) / 512 / 512 < 0.88:
    #     for i in range(1, k + 1):
    #         for j in range(m):
    #             print('……………………………………………………')
    #             print('inactive', np.sum(relabel_mask == INACTIVE) / 512 / 512)
    #             print('active', np.sum(relabel_mask == ACTIVE) / 512 / 512)
    #             print('passive', np.sum(relabel_mask == PASSIVE) / 512 / 512)
    #             print('')
    #             push[grid_shape, block_shape](d_weight_left, d_weight_up, d_weight_right, d_weight_down,
    #                                           d_pull_up, d_pull_right, d_pull_down, d_pull_left,
    #                                           d_weight_frg, d_weight_bkg, d_height, w, h, d_relabel_mask)
    #
    #             pull[grid_shape, block_shape](d_weight_left, d_weight_up, d_weight_right, d_weight_down,
    #                                           d_pull_up, d_pull_right, d_pull_down, d_pull_left,
    #                                           d_weight_frg, d_weight_bkg, d_height, d_relabel_mask, w, h)
    #
    #         local_relabel[grid_shape, block_shape](d_weight_left, d_weight_up, d_weight_right, d_weight_down,
    #                                                d_pull_up, d_pull_right, d_pull_down, d_pull_left,
    #                                                d_weight_frg, d_weight_bkg, d_height, d_relabel_mask, w, h)

    #     d_assigned.fill(0)
    #     for i in range(1, k + 1):
    #         global_relabel[grid_shape, block_shape](d_weight_left, d_weight_up, d_weight_right, d_weight_down,
    #                                                 d_pull_up, d_pull_right, d_pull_down, d_pull_left,
    #                                                 d_weight_frg, d_weight_bkg, d_height, d_relabel_mask, w, h, k,
    #                                                 d_assigned)

    h_terminate_condition = 0
    counter = 1
    d_count_blocks = np.zeros(1)
    d_count_blocks[0] = grid_shape[0] * grid_shape[1]
    d_finish = np.zeros(1, dtype=bool)
    d_stochastic = np.zeros((grid_shape[1], grid_shape[0]))
    s_weight_frg = np.zeros(d_weight_frg.shape)
    while h_terminate_condition != 2:
    # while np.sum(relabel_mask == INACTIVE) / h / w < 0.9:
    #     print('……………………………………………………')
        print('inactive', np.sum(relabel_mask == INACTIVE) / h / w, counter)
        if counter % 10 == 0:
            d_finish[0] = False
            kernel_push_stochastic1[grid_shape, block_shape](d_weight_frg, s_weight_frg, d_count_blocks, d_finish)

            if d_finish[0]:
                h_terminate_condition += 1

        if counter % 11 == 0:
            d_stochastic.fill(0)
            d_count_blocks[0] = 0
            kernel_push_stochastic2[grid_shape, block_shape](d_weight_frg, s_weight_frg, d_stochastic)
            d_count_blocks[0] = np.sum(d_stochastic == 1)

        push[grid_shape, block_shape](d_weight_left, d_weight_up, d_weight_right, d_weight_down,
                                      d_pull_up, d_pull_right, d_pull_down, d_pull_left,
                                      d_weight_frg, d_weight_bkg, d_height, w, h, d_relabel_mask)

        pull[grid_shape, block_shape](d_weight_left, d_weight_up, d_weight_right, d_weight_down,
                                      d_pull_up, d_pull_right, d_pull_down, d_pull_left,
                                      d_weight_frg, d_weight_bkg, d_height, d_relabel_mask, w, h)

        local_relabel[grid_shape, block_shape](d_weight_left, d_weight_up, d_weight_right, d_weight_down,
                                               d_pull_up, d_pull_right, d_pull_down, d_pull_left,
                                               d_weight_frg, d_weight_bkg, d_height, d_relabel_mask,
                                               w, h)

        counter += 1

    pixel_mask = np.ones(input_shape, dtype=bool)
    d_pixel_mask = pixel_mask
    from_frg_bfs_init[grid_shape, block_shape](d_weight_left, d_weight_up, d_weight_right, d_weight_down,
                                               d_pull_up, d_pull_right, d_pull_down, d_pull_left,
                                               d_weight_frg, d_weight_bkg, d_height, d_relabel_mask, w, h, k,
                                               d_assigned, d_pixel_mask)

    over = np.zeros(1, dtype=bool)
    count = 1
    while not over:
        from_frg_bfs[grid_shape, block_shape](d_weight_left, d_weight_up, d_weight_right, d_weight_down,
                                              d_pull_up, d_pull_right, d_pull_down, d_pull_left,
                                              d_weight_frg, d_weight_bkg, d_height, d_relabel_mask, w, h, k,
                                              d_assigned, d_pixel_mask, count, over)
        count += 1
    return d_height > 0

