import glob
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
import sys

version = 'oc'


def load_ann_and_trans(ann_floder):
    ann_files = glob.glob(ann_floder + "/*.txt")

    gt_bboxes = []
    for ann_file in tqdm(ann_files):

        if os.path.getsize(ann_file) == 0:
            continue

        with open(ann_file) as f:
            s = f.readlines()
            for si in s:
                bbox_info = si.split()
                poly = np.array(bbox_info[:8], dtype=np.float32)
                try:
                    x, y, w, h, a = poly2obb_np(poly, version)
                except:
                    continue

                gt_bboxes.append([w, h])

    return gt_bboxes


def average_iou(bboxes, anchors):
    """Calculates the Intersection over Union (IoU) between bounding boxes and
    anchors.

    Args:
    bboxes : Array of bounding boxes in [width, height] format.
    anchors : Array of aspect ratios [n, 2] format.

    Returns:
    avg_iou_perc : A Float value, average of IOU scores from each aspect ratio
    """
    intersection_width = np.minimum(anchors[:, [0]], bboxes[:, 0]).T
    intersection_height = np.minimum(anchors[:, [1]], bboxes[:, 1]).T

    if np.any(intersection_width == 0) or np.any(intersection_height == 0):
        raise ValueError("Some boxes have zero size.")

    intersection_area = intersection_width * intersection_height
    boxes_area = np.prod(bboxes, axis=1, keepdims=True)
    anchors_area = np.prod(anchors, axis=1, keepdims=True).T
    union_area = boxes_area + anchors_area - intersection_area
    avg_iou_perc = np.mean(np.max(intersection_area / union_area, axis=1)) * 100

    return avg_iou_perc


def kmeans_aspect_ratios(bboxes, kmeans_max_iter, num_aspect_ratios):
    """Calculate the centroid of bounding boxes clusters using Kmeans algorithm.

    Args:
    bboxes : Array of bounding boxes in [width, height] format.
    kmeans_max_iter : Maximum number of iterations to find centroids.
    num_aspect_ratios : Number of centroids to optimize kmeans.

    Returns:
    aspect_ratios : Centroids of cluster (optmised for dataset).
    avg_iou_prec : Average score of bboxes intersecting with new aspect ratios.
    """

    assert len(bboxes), "You must provide bounding boxes"

    # normalized_bboxes = bboxes / np.sqrt(bboxes.prod(axis=1, keepdims=True))
    normalized_bboxes = bboxes / 800

    # Using kmeans to find centroids of the width/height clusters
    kmeans = KMeans(
        n_clusters=num_aspect_ratios, init='k-means++', max_iter=kmeans_max_iter)
    predict = kmeans.fit_predict(X=normalized_bboxes)
    ar = kmeans.cluster_centers_

    assert len(ar), "Unable to find k-means centroid, try increasing kmeans_max_iter."

    avg_iou_perc = average_iou(normalized_bboxes, ar)

    if not np.isfinite(avg_iou_perc):
        sys.exit("Failed to get aspect ratios due to numerical errors in k-means")

    aspect_ratios = [w / h for w, h in ar]

    return normalized_bboxes, aspect_ratios, avg_iou_perc, ar, predict


if __name__ == '__main__':

    ann_files_train = 'E:/lrk/trail/datasets/DOTA-v1.5/divide_ship/train/annfiles/'
    ann_files_val = 'E:/lrk/trail/datasets/DOTA-v1.5/divide_ship/val/annfiles/'
    ann_files_test = 'E:/lrk/trail/datasets/DOTA-v1.5/divide_ship/test/annfiles/'

    gt_bboxes_np = []
    print("load train ann_files")
    train_booxes_np = load_ann_and_trans(ann_files_train)
    print("load train ann_files done")

    # print("load val ann_files")
    # val_booxes_np = load_ann_and_trans(ann_files_val)
    # print("load val ann_files done")

    # print("load test ann_files")
    # test_booxes_np = load_ann_and_trans(ann_files_test)
    # print("load test ann_files done")

    print("-----------------------------------------")
    print("train_shape: ", np.array(train_booxes_np).shape)
    # print("val_shape: ", np.array(val_booxes_np).shape)
    # print("test_shape: ", np.array(test_booxes_np).shape)

    gt_bboxes_np.extend(train_booxes_np)
    # gt_bboxes_np.extend(val_booxes_np)
    # gt_bboxes_np.extend(test_booxes_np)

    a = np.array(gt_bboxes_np)

    print("gt_bboxes_shape: ", np.array(gt_bboxes_np).shape)

    # ann_files = 'E:/lrk/kmeans++/txt/'
    # gt_bboxes_np = load_ann_and_trans(ann_files)
    # print("load txt and transfor done")

    path = 'E:/lrk/kmeans++/10/'
    path_cluster = path + 'cluster/'
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path_cluster):
        os.mkdir(path_cluster)

    cluster_num = []
    cluster_avgIou = []

    for i in tqdm(range(3, 11)):
        normalized_bboxes, aspect_ratios_i, avg_iou_perc_i, ar, predict = kmeans_aspect_ratios(
            bboxes=np.array(gt_bboxes_np),
            kmeans_max_iter=1000,
            num_aspect_ratios=i)

        # 长宽比写入文件
        cluster_num.append(i)
        cluster_avgIou.append(avg_iou_perc_i)

        cluster_path = path_cluster + str(i) + ".txt"
        f = open(cluster_path, 'w', encoding='utf-8')
        row = np.shape(aspect_ratios_i)[0]
        for j in range(row):
            ratio = '%.4f \n' % (aspect_ratios_i[j])
            f.write(ratio)
        f.close()

        # 画图显示样本数据
        plt.figure('Kmeans', facecolor='lightgray')
        plt.title('Kmeans', fontsize=16)
        plt.xlabel('w', fontsize=14)
        plt.ylabel('h', fontsize=14)
        plt.tick_params(labelsize=10)
        plt.scatter(np.array(normalized_bboxes)[:, 0], np.array(normalized_bboxes)[:, 1], s=80, c=predict, cmap='brg')

        plt.scatter(ar[:, 0], ar[:, 1], s=100, marker="*", c="black",
                    label="cluster center")
        plt.legend()
        plt.savefig(path + 'kmeans++_scatter_' + str(i) + '.jpg')
        plt.show()

        # anchors_i, _, avgIou_i = estimateAnchorBoxes(np.array(gt_bboxes_np), numAnchors=i)
        #
        # cluster_num.append(i)
        # cluster_avgIou.append(avgIou_i)
        #
        # cluster = anchors_i[np.argsort(anchors_i[:, 0])]
        #
        # cluster_path = 'E:/lrk/kmeans++/cluster/' + str(i) + ".txt"
        #
        # f = open(cluster_path, 'w', encoding='utf-8')
        # row = np.shape(cluster)[0]
        # for j in range(row):
        #     w_h = "%d, %d \n" % (cluster[j][0], cluster[j][1])
        #     f.write(w_h)
        # f.close()
    plt.title('Kmeans', fontsize=16)
    plt.plot(cluster_num, cluster_avgIou)
    plt.xlabel("cluster_num")
    plt.ylabel("avg_iou")
    plt.savefig(path + 'kmeans++_line.jpg')
    plt.show()
