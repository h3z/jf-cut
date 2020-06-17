import pickle
import time

import numpy as np
import cv2
from grabcut.jf_bfs.jf_bfs_seq.cfg import *
from grabcut.jf_bfs.jf_bfs_seq.computeCut import compute_cut
from grabcut.jf_bfs.jf_bfs_seq.computeTag import compute_tag
from grabcut.jf_bfs.jf_bfs_seq.countActiveBlock import count_active_block
from grabcut.jf_bfs.jf_bfs_seq.globalRelabel import global_relabel, global_relabel_jf_init
from grabcut.jf_bfs.jf_bfs_seq.jumpFloodBFS import compute_bfs
from grabcut.jf_bfs.jf_bfs_seq.localRelabelPush import relabel_push


def test():
    # node_excess = pickle.load(open('/home/hzzz/share/excesses.p', 'rb'))
    # node_capacity = pickle.load(open('/home/hzzz/share/capacity.p', 'rb'))
    # node_label_height = pickle.load(open('/home/hzzz/share/node_label_height.p', 'rb'))

    node_excess = pickle.load(open('/home/hzzz/jiubai_env/data/init_excess.p', 'rb'))
    node_capacity = pickle.load(open('/home/hzzz/jiubai_env/data/init_capacity.p', 'rb'))
    node_label_height = pickle.load(open('/home/hzzz/jiubai_env/data/global_label_height.p', 'rb'))

    flow_init = np.sum(node_excess[node_excess > 0])
    img_shape = node_excess.shape
    block_size_h = (img_shape[0] - 1) // BLOCK_SIZE + 1
    block_size_w = (img_shape[1] - 1) // BLOCK_SIZE + 1
    block_shape = (block_size_h, block_size_w)

    block_nearest_yx = np.zeros((block_shape[0], block_shape[1], 2), dtype=int)
    block_nearest_distance = np.zeros(block_shape, dtype=int)
    cut_data = np.zeros(img_shape, dtype=int)

    t1 = time.time()
    if not DEBUG:
        global_relabel_jf_init(block_nearest_yx, block_nearest_distance, node_excess)
        list_data, histogram_data, max_distance = compute_bfs(block_nearest_yx, block_nearest_distance, block_shape)
        # 45秒
        node_label_height = global_relabel(node_excess, node_capacity, list_data, histogram_data, max_distance)

    finished = False
    count = 1
    while not finished:
        count += 1
        list_data, block_offset = count_active_block(node_excess, node_label_height)
        print('block_offset ', block_offset)
        relabel_push(list_data, block_offset, node_excess, node_label_height, node_capacity)

        finished = compute_cut(block_nearest_yx, block_nearest_distance, node_capacity, node_excess, cut_data,
                               block_shape)

        print(flow_init - np.sum(node_excess[node_excess > 0]))

    print(count)

    # node_excess = pickle.load(open('/home/hzzz/jiubai_env/data/final_node_excess.p', 'rb'))
    # node_capacity = pickle.load(open('/home/hzzz/jiubai_env/data/final_node_capacity.p', 'rb'))
    # cut_data = pickle.load(open('/home/hzzz/jiubai_env/data/final_cut_data.p', 'rb'))
    # node_label_height = pickle.load(open('/home/hzzz/jiubai_env/data/final_node_label_height.p', 'rb'))
    #
    pickle.dump(node_excess, open('/home/hzzz/jiubai_env/data/final_node_excess.p', 'wb'))
    pickle.dump(node_capacity, open('/home/hzzz/jiubai_env/data/final_node_capacity.p', 'wb'))
    pickle.dump(cut_data, open('/home/hzzz/jiubai_env/data/final_cut_data.p', 'wb'))
    pickle.dump(node_label_height, open('/home/hzzz/jiubai_env/data/final_node_label_height.p', 'wb'))

    tag_data = np.zeros(img_shape, dtype=int)
    compute_tag(block_nearest_yx, block_nearest_distance, node_excess, cut_data, tag_data, block_shape)
    pickle.dump(tag_data, open('/home/hzzz/jiubai_env/data/final_tag.p', 'wb'))
    print('time: ', time.time() - t1)

    img = cv2.imread('/home/hzzz/jiubai_env/img/lena.jpg')
    img[tag_data == 0] = 0
    cv2.imwrite('/home/hzzz/jiubai_env/output/test_1.jpg', img)

if __name__ == '__main__':
    test()
