import math
import pickle
import time
from copy import deepcopy

import cv2
import maxflow
import numba
import numpy as np
# from kmeans import kmeans_v1 as hzzz_kmeans
from cv2 import kmeans
from numba import cuda
from numba.typed import List
from numpy import logical_or, logical_and, logical_not

""" 
model: 5 * 13 = 65 
0 - 4 -> k1 - k5
5 - 20 -> μ1 - μ5
20 - 65 -> ∑1 - ∑5  (逆)
"""

EPS = 2.220446049250313e-16
FRG = 1
BKG = 0
PR_BKG = 3
PR_FRG = 4

N = 5
# 别问，问就是抄的丹方。
BORDER_LOSS_WEIGHT = 50  # gamma
DONT_CUT_ME = 9 * BORDER_LOSS_WEIGHT


def min_cut(img, bkg_similar, frg_similar, around_capacity):
    pickle.dump((frg_similar - bkg_similar), open('/home/hzzz/share/excesses.p', 'wb'))
    shape = bkg_similar.shape
    capacity = np.zeros((shape[0], shape[1], 4))
    capacity[..., 0] = around_capacity['up']
    capacity[: -1, :, 1] = around_capacity['up'][1:, :]
    capacity[..., 2] = around_capacity['left']
    capacity[:, :-1, 3] = around_capacity['left'][:, 1:]
    pickle.dump(capacity, open('/home/hzzz/share/capacity.p', 'wb'))

    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes(img.shape[:2])
    g.add_grid_tedges(nodeids, bkg_similar, frg_similar)

    left = around_capacity['left']
    g.add_grid_edges(nodeids, left, np.array([[0, 0, 0],
                                              [1, 0, 0],
                                              [0, 0, 0]]), symmetric=True)

    upleft = around_capacity['upleft']
    g.add_grid_edges(nodeids, upleft, np.array([[1, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0]]), symmetric=True)

    up = around_capacity['up']
    g.add_grid_edges(nodeids, up, np.array([[0, 1, 0],
                                            [0, 0, 0],
                                            [0, 0, 0]]), symmetric=True)
    upright = around_capacity['upright']
    g.add_grid_edges(nodeids, upright, np.array([[0, 0, 1],
                                                 [0, 0, 0],
                                                 [0, 0, 0]]), symmetric=True)

    t = time.time()
    g.maxflow()
    print('min cut time: ', time.time() - t)

    return g.get_grid_segments(nodeids)


def init_capacity_around(img):
    shape = img.shape[:2]
    count = 4 * shape[0] * shape[1] - 3 * (shape[0] + shape[1]) + 2

    diff = img[:, 1:] - img[:, :-1]
    left_similar = np.zeros(shape)
    left_similar[:, 1:] = np.sum(np.square(diff), axis=-1)

    diff = img[1:, 1:] - img[:-1, :-1]
    upleft_similar = np.zeros(shape)
    upleft_similar[1:, 1:] = np.sum(np.square(diff), axis=-1)

    diff = img[1:] - img[:-1]
    up_similar = np.zeros(shape)
    up_similar[1:] = np.sum(np.square(diff), axis=-1)

    diff = img[1:, :-1] - img[:-1, 1:]
    upright_similar = np.zeros(shape)
    upright_similar[1:, :-1] = np.sum(np.square(diff), axis=-1)

    beta = np.sum(left_similar) + np.sum(upleft_similar) + np.sum(up_similar) + np.sum(upright_similar)
    beta = - count / beta / 2.

    left = np.exp(beta * left_similar)
    left[:, 0] = 0
    up = np.exp(beta * up_similar)
    up[0, :] = 0
    upleft = np.exp(beta * upleft_similar)
    upleft[:, 0] = 0
    upleft[0, :] = 0
    upright = np.exp(beta * upright_similar)
    upright[:, -1] = 0
    upright[0, :] = 0

    return {'left': BORDER_LOSS_WEIGHT * left,
            'up': BORDER_LOSS_WEIGHT * up,
            'upleft': BORDER_LOSS_WEIGHT * upleft / np.sqrt(2),
            'upright': BORDER_LOSS_WEIGHT * upright / np.sqrt(2)}


@numba.jit
def gmm_predict(gmm, pixel):
    res = 0.
    for i in range(N):
        res += gaussian_predict(gmm, i, pixel)
    return res


@numba.jit
def gaussian_predict(gmm, i, pixel):
    sigma = 4 * N + 9 * i
    miu = N + 3 * i
    d0 = pixel[0] - gmm[miu]
    d1 = pixel[1] - gmm[miu + 1]
    d2 = pixel[2] - gmm[miu + 2]

    dis = -(d0 * (d1 * gmm[sigma + 1] +
                  d2 * gmm[sigma + 2]) +
            d1 * d2 * gmm[sigma + 5] +
            (d0 * d0 * gmm[sigma] +
             d1 * d1 * gmm[sigma + 4] +
             d2 * d2 * gmm[sigma + 8]) / 2)
    return gmm[i] * math.exp(dis)


gaussian_predict_gpu = cuda.jit(device=True)(gaussian_predict)


def which_gaussian(gmm, pixel):
    best = 0
    maxv = 0

    for i in range(N):
        v = gaussian_predict_gpu(gmm, i, pixel)
        if v > maxv:
            maxv = v
            best = i
    return best


which_gaussian_gpu = cuda.jit(device=True)(which_gaussian)


@cuda.jit
def EM_E_gmm(gmm, pixels, labels):
    i = cuda.grid(1)
    labels[i] = which_gaussian_gpu(gmm, pixels[i])


def group_by(pixels, labels):
    res = List()
    for i in range(N):
        res.append(pixels[labels == i])
    return res


def empty_2d_list(type_by_data):
    lst = List()
    for i in range(N):
        sub = List()
        sub.append(type_by_data)
        sub.pop()
        lst.append(sub)
    return lst


def f(gmm, pixels):
    l = pixels.shape[0]
    labels = np.zeros(l)
    d_pixels = cuda.to_device(pixels)
    d_labels = cuda.to_device(labels)
    d_gmm = cuda.to_device(gmm)
    EM_E_gmm[(l + 64) // 64, 64](d_gmm, d_pixels, d_labels)
    d_labels.to_host()
    return labels


def EM_E(img, mask, bkg_gmm, frg_gmm):
    bkg_pixels = img[logical_or(mask == BKG, mask == PR_BKG)]
    bkg_labels = f(bkg_gmm, bkg_pixels)

    frg_pixels = img[logical_or(mask == FRG, mask == PR_FRG)]
    frg_labels = f(frg_gmm, frg_pixels)

    return group_by(bkg_pixels, np.array(bkg_labels)), group_by(frg_pixels, np.array(frg_labels))


def check_cov(cov, cov_det):
    if cov_det < np.finfo(float).eps:
        print('oh  det is zero?')
        cov[:, 0] += 0.01
        cov_det = np.linalg.det(cov)
    assert cov_det > np.finfo(float).eps


def EM_M(gmm, pixels):
    total = 0
    for i in range(N):
        x = pixels[i]
        l = len(x)

        miu = np.mean(x, 0)  # 3,
        d = x - miu  # large * 3
        cov = np.dot(d.T, d) / l  # 3,3
        cov_det = np.linalg.det(cov)  # 1
        check_cov(cov, cov_det)

        gmm[i] = 1. / math.sqrt(cov_det)
        gmm[N + i * 3:N + (i + 1) * 3] = miu
        gmm[4 * N + i * 9:4 * N + (i + 1) * 9] = np.linalg.inv(cov).reshape(9)
        total += l

    for i in range(N):
        gmm[i] *= len(pixels[i]) / total


def rect_marker(shape, rect):
    i1, j1, i2, j2 = rect
    mini, maxi = min(i1, i2), max(i1, i2)
    minj, maxj = min(j1, j2), max(j1, j2)
    mask = np.zeros(shape)
    mask[mini:maxi, minj:maxj] = PR_FRG
    mask[:mini, :] = BKG
    mask[maxi:, :] = BKG
    mask[:, :minj] = BKG
    mask[:, maxj:] = BKG
    return mask


# 通过最小割，更新mask
@numba.jit
def compute_st_edge(img, mask, bkg_gmm, frg_gmm):
    shape = img.shape[:2]
    bkg_similar = np.zeros(shape)
    frg_similar = np.zeros(shape)

    for i in range(shape[0]):
        for j in range(shape[1]):
            cur = (i, j)
            pixel = img[cur]
            if mask[cur] == BKG:
                bkg_similar[cur], frg_similar[cur] = DONT_CUT_ME, 0
            elif mask[cur] == FRG:
                bkg_similar[cur], frg_similar[cur] = 0, DONT_CUT_ME
            else:
                frg_similar[cur] = -np.log(gmm_predict(bkg_gmm, pixel))
                bkg_similar[cur] = -np.log(gmm_predict(frg_gmm, pixel))
    return bkg_similar, frg_similar


def init_model_by_kmeans(pixels):
    _, labels, _ = kmeans(pixels.astype(np.float32), N, None,
                          (cv2.TERM_CRITERIA_MAX_ITER, 10, 0), 0,
                          cv2.KMEANS_PP_CENTERS)
    labels = np.squeeze(labels)

    return group_by(pixels, labels)


def grabCut(img, mask, rect, bkg_gmm, frg_gmm, iter_count, mode):
    img = img.astype(np.float32)
    x, y, w, h = rect
    rect = (y, x, y + h, x + w)
    if mode == cv2.GC_INIT_WITH_RECT:
        mask = rect_marker(img.shape[:2], rect).astype(np.float32)

    around_capacity = init_capacity_around(img)

    bkg_gmm = np.zeros(N * 13).astype(np.float32)  # if bkg_gmm is None else bkg_gmm
    frg_gmm = np.zeros(N * 13).astype(np.float32)  # if frg_gmm is None else frg_gmm

    bkg_group_by = init_model_by_kmeans(img[logical_or(mask == BKG, mask == PR_BKG)])
    frg_group_by = init_model_by_kmeans(img[logical_or(mask == FRG, mask == PR_FRG)])
    EM_M(bkg_gmm, bkg_group_by)
    EM_M(frg_gmm, frg_group_by)

    for i in range(iter_count):
        bkg_group_by, frg_group_by = EM_E(img, mask, bkg_gmm, frg_gmm)

        EM_M(bkg_gmm, bkg_group_by)
        EM_M(frg_gmm, frg_group_by)

        # min-cut
        bkg_similar, frg_similar = compute_st_edge(img, mask, bkg_gmm, frg_gmm)
        sgm = min_cut(img, bkg_similar, frg_similar, around_capacity)

        mask[logical_and(mask != FRG, sgm)] = PR_FRG
        mask[logical_and(mask != BKG, logical_not(sgm))] = PR_BKG

    return logical_or(mask == FRG, mask == PR_FRG)


def test():
    print('update3')
    img = cv2.imread('/home/hzzz/code/imgs/lena.jpg')
    rect = (39, 39, 394, 457)
    img2 = deepcopy(img)
    # rect = (362, 512, 1088, 1988)
    mask = None
    for i in range(2):
        t = time.time()
        mask = grabCut(img, None, rect, None, None, 1, cv2.GC_INIT_WITH_RECT)
        print('time: ', time.time() - t)

    img2[mask == False] = 0
    cv2.imwrite('/home/hzzz/code/imgs/release_pymaxflow/output.jpg', img2)
    return img2


if __name__ == '__main__':
    test()
