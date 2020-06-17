import pickle
import time

import cv2

from grabcut.jf_bfs.jf_bfs_cuda.computeCut import cuda_compute_cut
from grabcut.jf_bfs.jf_bfs_cuda.computeTag import cuda_compute_tag
from grabcut.jf_bfs.jf_bfs_cuda.countActiveBlock import cuda_count_active_block
from grabcut.jf_bfs.jf_bfs_cuda.globalRelabel import *
from grabcut.jf_bfs.jf_bfs_cuda.jumpFloodBFS import cuda_compute_bfs
from grabcut.jf_bfs.jf_bfs_cuda.localRelabelPush import cuda_relabel_push


def log_time(s, t):
    print(s + ' time: ', time.time() - t)
    return time.time()


def test():
    # node_excess = pickle.load(open('/home/hzzz/share/excesses.p', 'rb'))
    # node_capacity = pickle.load(open('/home/hzzz/share/capacity.p', 'rb'))
    # node_label_height = pickle.load(open('/home/hzzz/share/node_label_height.p', 'rb'))

    node_excess = pickle.load(open('/home/hzzz/jiubai_env/data/init_excess.p', 'rb'))
    node_capacity = pickle.load(open('/home/hzzz/jiubai_env/data/init_capacity.p', 'rb'))
    node_label_height_target = pickle.load(open('/home/hzzz/jiubai_env/data/global_label_height.p', 'rb'))

    flow_init = np.sum(node_excess[node_excess > 0])
    img_shape = node_excess.shape
    block_size_h = (img_shape[0] - 1) // BLOCK_SIZE + 1
    block_size_w = (img_shape[1] - 1) // BLOCK_SIZE + 1
    block_shape = (block_size_h, block_size_w)

    block_nearest_yx = np.zeros((block_shape[0], block_shape[1], 2), dtype=int)
    block_nearest_distance = np.zeros(block_shape, dtype=int)
    cut_data = np.zeros(img_shape, dtype=int)
    node_label_height = np.zeros(img_shape, dtype=int)

    t1 = time.time()

    cuda_global_relabel_jf_init(block_nearest_yx, block_nearest_distance, node_excess, node_label_height)
    list_data, histogram_data, max_distance = cuda_compute_bfs(block_nearest_yx, block_nearest_distance, block_shape)
    # 45秒 -> cuda ?， 没测，应该快吧
    cuda_global_relabel(node_capacity, node_label_height, list_data, histogram_data, max_distance)

    t1 = log_time('global relabel', t1)
    begin = t1
    count_active_block_time = 0
    local_push_relabel_time = 0
    compute_cut_time = 0
    finished = False
    count = -1
    block_offset = None
    while not finished:
        count += 1
        if count % 8 == 0:
            list_data, block_offset = cuda_count_active_block(node_excess, node_label_height)
            count_active_block_time += (time.time() - t1)
            t1 = time.time()
        cuda_relabel_push(list_data, block_offset, node_excess, node_label_height, node_capacity)
        # log_time('first, ', t1)
        local_push_relabel_time += (time.time() - t1)
        t1 = time.time()
        if count % 8 == 0:
            finished = cuda_compute_cut(block_nearest_yx, block_nearest_distance, node_capacity, node_excess, cut_data,
                                        block_shape)
            compute_cut_time += (time.time() - t1)
            t1 = time.time()

        print(int(flow_init - np.sum(node_excess[node_excess > 0])))

    print(count)

    tag_data = np.zeros(img_shape, dtype=int)
    cuda_compute_tag(block_nearest_yx, block_nearest_distance, node_excess, cut_data, tag_data, block_shape)
    compute_cut_time += (time.time() - t1)

    print('total time: ', time.time() - begin)
    print('count active block time', count_active_block_time)
    print('local push relabel time', local_push_relabel_time)
    print('compute cut time', compute_cut_time)

    pickle.dump(tag_data, open('/home/hzzz/jiubai_env/data/final_tag.p', 'wb'))
    img = cv2.imread('/home/hzzz/jiubai_env/img/lena.jpg')
    img[tag_data == 0] = 0
    cv2.imwrite('/home/hzzz/jiubai_env/output/debug_616/result11.jpg', img)


if __name__ == '__main__':
    test()
