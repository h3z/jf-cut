import math
import time
from copy import deepcopy
import pickle
import cv2
import maxflow
import numba
import numpy as np
# from kmeans import kmeans_v1 as hzzz_kmeans
from cv2 import kmeans
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
# 悟了，9是因为8条边+st边。 意思是，为了你可以放弃全世界
DONT_CUT_ME = 9 * BORDER_LOSS_WEIGHT


def min_cut(img, bkg_similar, frg_similar, around_capacity):
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

    v = g.maxflow()
    print(v)
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
    print(beta)

    return {'left': BORDER_LOSS_WEIGHT * np.exp(beta * left_similar),
            'up': BORDER_LOSS_WEIGHT * np.exp(beta * up_similar),
            'upleft': BORDER_LOSS_WEIGHT * np.exp(beta * upleft_similar) / np.sqrt(2),
            'upright': BORDER_LOSS_WEIGHT * np.exp(beta * upright_similar) / np.sqrt(2)}


@numba.njit
def gmm_predict(gmm, pixel):
    res = 0.
    for i in range(N):
        res += gaussian_predict(gmm, i, pixel)
    return res


@numba.njit
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


@numba.njit
def gaussian_predict_guess(gmm, i, pixel):
    miu = N + 3 * i
    d0 = pixel[0] - gmm[miu]
    d1 = pixel[1] - gmm[miu + 1]
    d2 = pixel[2] - gmm[miu + 2]
    return (d0 ** 2 + d1 ** 2 + d2 ** 2)


@numba.njit
def which_gaussian(gmm, pixel):
    best = 0
    maxv = 0

    for i in range(N):
        v = gaussian_predict(gmm, i, pixel)
        if v > maxv:
            maxv = v
            best = i
    return best


@numba.njit
def EM_E_gmm(gmm, pixels):
    labels = List()
    for pixel in pixels:
        labels.append(which_gaussian(gmm, pixel))
    return labels


def group_by(pixels, labels):
    res = List()
    for i in range(N):
        res.append(pixels[labels == i])
    return res


def EM_E(img, mask, bkg_gmm, frg_gmm):
    bkg_pixels = img[logical_or(mask == BKG, mask == PR_BKG)]
    bkg_labels = EM_E_gmm(bkg_gmm, bkg_pixels)

    frg_pixels = img[logical_or(mask == FRG, mask == PR_FRG)]
    frg_labels = EM_E_gmm(frg_gmm, frg_pixels)

    return group_by(bkg_pixels, np.array(bkg_labels)), group_by(frg_pixels, np.array(frg_labels))


def check_cov(cov, cov_det):
    if cov_det < np.finfo(float).eps:
        print('oh  det is zero?')
        cov[:, 0] += 0.01
        cov_det = np.linalg.det(cov)
    assert cov_det > np.finfo(float).eps


def some(M):
    # M: 3,3
    U, Sigma, VT = np.linalg.svd(M)
    ttt = np.sqrt(Sigma)
    return np.zeros(3)


def generate_cache(miu, cov_I):
    # TODO
    A = some(cov_I)  # A * A.T = cov_I
    column_times_cache = List()
    for c in range(3):
        column = A[:, c]  # 3,
        cache = List()
        for i in range(256):
            cache.append(column * i)
        column_times_cache.append(cache)
    return column_times_cache


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
        # generate_cache(miu, np.linalg.inv(cov))
        total += l

        # rate = 0  # 落在区间内的比例（越大越值得搞）
        # rangee = 0  # 区间大小（越大需要缓存越多。 立方级别）
        # nn = 3  # 4个sigma
        #
        # d = d[d[:, 0] < + nn * math.sqrt(cov[0, 0])]
        # print(len(d), l)
        # d = d[d[:, 0] > - nn * math.sqrt(cov[0, 0])]
        # print(len(d), l)
        # d = d[d[:, 1] < + nn * math.sqrt(cov[1, 1])]
        # print(len(d), l)
        # d = d[d[:, 1] < + nn * math.sqrt(cov[1, 1])]
        # print(len(d), l)
        # d = d[d[:, 2] < + nn * math.sqrt(cov[2, 2])]
        # print(len(d), l)
        # d = d[d[:, 2] < + nn * math.sqrt(cov[2, 2])]
        # print(len(d), l)
        #
        # rate = len(d) / l
        # rangee = np.max(d, 0) - np.min(d, 0)
        # print(rate, rangee)
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
@numba.njit
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
        mask = rect_marker(img.shape[:2], rect)

    around_capacity = init_capacity_around(img)

    bkg_gmm = np.zeros(N * 13).astype(np.float32)  # if bkg_gmm is None else bkg_gmm
    frg_gmm = np.zeros(N * 13).astype(np.float32)  # if frg_gmm is None else frg_gmm
    # for i in range(N * 13):
    #     bkg_gmm.append(0.)
    #     frg_gmm.append(0.)

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
    print('update2')
    img = cv2.imread('/home/hzzz/code/imgs/lena.jpg')
    rect = (39, 39, 394, 457)
    # rect = (362, 512, 1088, 1988)
    img2 = deepcopy(img)
    mask = None
    for i in range(2):
        t = time.time()
        mask = grabCut(img, None, rect, None, None, 1, cv2.GC_INIT_WITH_RECT)
        print('time: ', time.time() - t)
    img2[mask == False] = 0
    return img2



if __name__ == '__main__':
    test()
