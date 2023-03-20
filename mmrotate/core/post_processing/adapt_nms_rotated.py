import torch
import numpy as np
import math
from mmrotate.core.bbox.iou_calculators import rbbox_overlaps


def adapt_nms_rotated_(m_bboxes,
                       m_scores,
                       score_thr,
                       nms,
                       max_num=-1,
                       score_factors=None,
                       return_inds=False):
    """

     Args:
        m_bboxes (torch.Tensor): shape (n, #class*5) or (n, 5)
        m_scores (torch.Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms (float): Config of NMS.
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.

    Returns:
        tuple (dets, labels, indices (optional)): tensors of shape (k, 5), \
        (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """

    num_classes = m_scores.size(1) - 1

    if m_bboxes.shape[1] > 5:
        bboxes = m_bboxes.view(m_scores.size(0), -1, 5)
    else:
        bboxes = m_bboxes[:, None].expand(m_scores.size(0), num_classes, 5)

    # print(bboxes.shape)
    scores = m_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 5)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    # 移除低分框
    valid_mask = scores > score_thr
    if score_factors is not None:
        # 扩展形状以匹配原始形状的分数
        score_factors = score_factors.view(-1, 1).expand(m_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]

    if bboxes.numel() == 0:
        dets = torch.cat([bboxes, scores[:, None]], -1)
        if return_inds:
            return dets, labels, inds
        else:
            return dets, labels

    max_coordinate = bboxes[:, :2].max() + bboxes[:, 2:4].max()
    offsets = labels.to(bboxes) * (max_coordinate + 1)
    if bboxes.size(-1) == 5:
        bboxes_for_nms = bboxes.clone()
        bboxes_for_nms[:, :2] = bboxes_for_nms[:, :2] + offsets[:, None]
    else:
        bboxes_for_nms = bboxes + offsets[:, None]

    # print(bboxes_for_nms.shape)
    # print(scores.shape)

    dets_wl = bboxes_for_nms
    _, order = scores.sort(0, descending=True)
    dets_sorted = dets_wl.index_select(0, order)
    # print(dets_sorted.shape)
    # ------------------------------------------------------------- #
    #   dets_wl: 原始未排序目标[N, 5]
    #   scores: 未排序分数[N, ] tensor([0.8400, 0.7300, 0.6700, 0.5800, 0.9100, 0.9400, 0.8000, 0.4800, 0.8000,
    #         0.8600, 0.5900, 0.1800, 0.1000, 0.6300, 0.8200, 0.7600, 0.8300, 0.2100,
    #         0.1800, 0.7300])
    #   order: 按照分数降序排序后的下标 tensor([ 5,  4,  9,  0, 16, 14,  6,  8, 15,  1, 19,  2, 13, 10,  3,  7, 17, 11,
    #         18, 12])
    #   dets_sorted: 排序后的目标[N, 5]
    # ------------------------------------------------------------- #

    keep = density_nms(bboxes_for_nms, scores, order, dets_sorted, nms.iou_thr, nms.sigma)
    # keep = density_nms(bboxes_for_nms, scores, order, dets_sorted)

    if max_num > 0:
        keep = keep[:max_num]

    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if return_inds:
        return torch.cat([bboxes, scores[:, None]], 1), labels, keep
    else:
        return torch.cat([bboxes, scores[:, None]], 1), labels


def density_nms(dets_wl, scores, order, dets_sorted, iou_thr=0.5, sigma=1):
    """
    #   dets_wl: 原始未排序目标[N, 5]
    #   scores: 未排序分数[N, ] tensor([0.8400, 0.7300, 0.6700, 0.5800, 0.9100, 0.9400, 0.8000, 0.4800, 0.8000,
    #         0.8600, 0.5900, 0.1800, 0.1000, 0.6300, 0.8200, 0.7600, 0.8300, 0.2100,
    #         0.1800, 0.7300])
    #   order: 按照分数降序排序后的下标 tensor([ 5,  4,  9,  0, 16, 14,  6,  8, 15,  1, 19,  2, 13, 10,  3,  7, 17, 11,
    #         18, 12])
    #   dets_sorted: 排序后的目标[N, 5]
    Args:
        dets_wl: 始未排序目标[N, 5]
        scores: 未排序分数[N, ]
        order: 按照分数降序排序后的下标
        dets_sorted: 排序后的目标[N, 5]
        iou_thr: IOU阈值
        sigma: 平衡调整参数

    Returns:

    """

    keep = []

    while dets_sorted.shape[0] > 0:

        keep.append(order[0].item())

        if dets_sorted.shape[0] == 1:
            break

        ious, num = rbbox_intersection(dets_sorted[0], dets_sorted[1:])

        # 计算密集度和自适应阈值
        # ------------------------------------------------------------------------ #
        # density = (num / (dets_sorted.shape[0] - 1)) * sigma
        # threshold = iou_thr * np.exp(-density)
        # threshold = iou_thr * np.tanh(density)
        # ------------------------------------------------------------------------ #

        density = (num / (dets_sorted.shape[0] - 1)) * sigma
        threshold = iou_thr * density

        # ------------------------------------------------------------------------ #

        mask = ious < threshold

        dets_sorted = dets_sorted[1:]
        order = order[1:]

        dets_sorted = dets_sorted[mask]
        order = order[mask]

    return torch.tensor(keep, dtype=torch.long)


def rbbox_intersection(current_rbox, other_rboxs):
    """
    计算当前框与其余框的有重叠的数量
    Args:
        current_rbox: 当前框  (5, )     (x_c, y_c, w, h, θ)
        other_rboxs:  其余框  (n, 5)

    Returns:
        iou_d_calculator: 当前框与其他框的IOU
        num: 当前框与其他框有交集的数量

    """

    current_rbox = current_rbox.reshape(-1, 5)

    # print("current_box.shape: ", current_rbox)
    # print("other_bboxs.shape: ", other_rboxs.shape)

    iou_d_calculator = rbbox_overlaps(current_rbox, other_rboxs)

    # print(iou_d_calculator)
    # print(iou_d_calculator.shape)

    to = (iou_d_calculator > 0.0).to(torch.int32)
    num = to.sum().item()
    # print(num)
    return iou_d_calculator[0], num


if __name__ == '__main__':
    s_boxes = torch.tensor([[200, 200, 100, 100, math.pi / 2],
                            [220, 220, 120, 220, math.pi],
                            [200, 220, 300, 140, 0],
                            [240, 200, 440, 400, math.pi / 3],
                            [210, 224, 156, 207, math.pi / 3],
                            [220, 226, 126, 106, math.pi / 4],
                            [205, 215, 104, 99, math.pi / 2],
                            [189, 204, 139, 180, math.pi / 8],
                            [178, 168, 379, 415, math.pi / 6],
                            [159, 157, 399, 410, math.pi / 7],
                            [176, 192, 430, 459, math.pi / 8],
                            [208, 217, 426, 444, math.pi / 6],
                            [14, 15, 27, 30, math.pi / 7],
                            [13.5, 17, 26, 35, math.pi / 3],
                            [12, 15, 27, 31, math.pi / 2],
                            [13, 14, 26, 30, math.pi / 4],
                            [1.5, 1.4, 2.1, 2.5, math.pi / 2],
                            [1.1, 1.2, 2.4, 2.5, math.pi / 4],
                            [1.1, 1.1, 2.4, 2.3, math.pi / 7],
                            [1, 1, 2, 2, math.pi / 6]], dtype=torch.float)
    s_scores = torch.tensor(
        [[0.84, 0.2], [0.73, 0.21], [0.67, 0.24], [0.58, 0.42], [0.91, 0.08], [0.94, 0.06], [0.8, 0.2], [0.48, 0.52], [0.8, 0.2],
         [0.86, 0.14],
         [0.59, 0.41], [0.18, 0.82], [0.1, 0.9], [0.63, 0.37], [0.82, 0.18], [0.76, 0.24], [0.83, 0.17], [0.21, 0.78], [0.18, 0.82],
         [0.73, 0.27]], dtype=torch.float)
    # print(s_boxes.shape)
    # print(s_scores.shape)
    # print(s_scores.size())
    nms = dict(iou_thr=0.5, sigma=1)
    adapt_nms_rotated_(s_boxes, s_scores, 0.05, nms)

    # rbbox_intersection(s_boxes[0], s_boxes[1:])
