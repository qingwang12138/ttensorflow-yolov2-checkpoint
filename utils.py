"""
Help functions for YOLOv2
"""
import random
import colorsys

import cv2
import numpy as np

# ############# preprocess image ##################


def preprocess_image(image, image_size=(416, 416)):
    # 输入image 和 image的尺寸
    # np.copy：以float32数据类型复制image数据
    image_cp = np.copy(image).astype(np.float32)
    # cv2.cvtColor:将image_cp图形由BGR转变为RGB
    image_rgb = cv2.cvtColor(image_cp, cv2.COLOR_BGR2RGB)
    # 将 image_rgb图像shape为416*416大小
    image_resized = cv2.resize(image_rgb, image_size)
    # normalize
    image_normalized = image_resized.astype(np.float32) / 255.0
    # 扩展batch_size维度
    image_expanded = np.expand_dims(image_normalized, axis=0)
    return image_expanded


def postprocess(bboxes, obj_probs, class_probs, image_shape=(416, 416),
                threshold=0.5):
    """post process the detection results"""
    bboxes = np.reshape(bboxes, [-1, 4])
    # bboxe的数据结构为[-1,[x,y,w,h]], x和w乘以图像列数， 将图像放在坐标轴上，图像的行数为y,列数为x
    bboxes[:, 0::2] *= float(image_shape[1])
    # y和h乘以图形行数
    bboxes[:, 1::2] *= float(image_shape[0])
    # 取整，预测框都为整数index值
    bboxes = bboxes.astype(np.int32)

    # clip the bboxs
    # 修改索引值，下标从0
    bbox_ref = [0, 0, image_shape[1] - 1, image_shape[0] - 1]
    # 剪辑边界框，不超出图像大小
    bboxes = bboxes_clip(bbox_ref, bboxes)
    # 计算13*13*5=845个先验框是否有目标的概率，数据格式是1*845
    obj_probs = np.reshape(obj_probs, [-1])

    # 计算845个先验框中属于某一类的概率，每个先验框中有80个类，所以数据结构是80*845的矩阵
    class_probs = np.reshape(class_probs, [len(obj_probs), -1])

    # np.argmax:返回沿着axis轴的最大索引值,只返回第一次出现最大索引值, axis为1是矩阵的列,axis为0是行
    # 就是在找着845个框中那一个框分别属于第0...79个类最大的可能性
    # 输出结构为1*845
    class_inds = np.argmax(class_probs, axis=1)
    # class_probs的举证为845*845
    class_probs = class_probs[np.arange(len(obj_probs)), class_inds]
    # 845个值
    scores = obj_probs * class_probs

    # 大于阈值的保留，阈值设置为0.5
    keep_inds = scores > threshold
    # 分别是先验框的索引、分数值的索引和类别索引
    bboxes = bboxes[keep_inds]
    scores = scores[keep_inds]
    class_inds = class_inds[keep_inds]

    # 排序找到top400个最大值
    class_inds, scores, bboxes = bboxes_sort(class_inds, scores, bboxes)
    # nms：非最大值抑制操作
    class_inds, scores, bboxes = bboxes_nms(class_inds, scores, bboxes)

    return bboxes, scores, class_inds


def draw_detection(im, bboxes, scores, cls_inds, labels, thr=0.3):
    # draw_detection(image, bboxes, scores, class_inds, class_names)
    # Generate colors for drawing bounding boxes.

    hsv_tuples = [(x / float(len(labels)), 1., 1.)
                  for x in range(len(labels))]
    # lambda：匿名函数，类似C中的宏
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    # 画图,这里的imgcv是原始图像
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    print(bboxes)
    print("############")
    for i, box in enumerate(bboxes):
        if scores[i] < thr:
            continue
        cls_indx = cls_inds[i]

        thick = int((h + w) / 300)
        # (box[0],box[1])是左上角坐标，(box[2],box[3])是右下角坐标， thick是所画线的宽度
        cv2.rectangle(imgcv,
                      (box[0], box[1]), (box[2], box[3]),
                      colors[cls_indx], thick)
        # 框上要标注的信息
        mess = '%s: %.3f: (%d, %d, %d, %d)' % (labels[cls_indx], scores[i], box[0], box[1], box[2], box[3])
        # 标注位置
        if box[1] < 20:
            text_loc = (box[0] + 2, box[1] + 15)
        else:
            text_loc = (box[0], box[1] - 10)
        # cv2.putText(照片，文字，位置，字体，字体大小，颜色，字体粗细)
        cv2.putText(imgcv, mess, text_loc,
                    cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * h, colors[cls_indx], thick // 3)

    return imgcv


############## process bboxes ##################
def bboxes_clip(bbox_ref, bboxes):
    # 相对于bbox_ref剪辑预测框
    bboxes = np.copy(bboxes)
    # 转置bboxes
    bboxes = np.transpose(bboxes)
    bbox_ref = np.transpose(bbox_ref)
    bboxes[0] = np.maximum(bboxes[0], bbox_ref[0])
    bboxes[1] = np.maximum(bboxes[1], bbox_ref[1])
    bboxes[2] = np.minimum(bboxes[2], bbox_ref[2])
    bboxes[3] = np.minimum(bboxes[3], bbox_ref[3])
    bboxes = np.transpose(bboxes)
    return bboxes


def bboxes_sort(classes, scores, bboxes, top_k=400):
    """Sort bounding boxes by decreasing order and keep only the top_k
    """
    # if priority_inside:
    #     inside = (bboxes[:, 0] > margin) & (bboxes[:, 1] > margin) & \
    #         (bboxes[:, 2] < 1-margin) & (bboxes[:, 3] < 1-margin)
    #     idxes = np.argsort(-scores)
    #     inside = inside[idxes]
    #     idxes = np.concatenate([idxes[inside], idxes[~inside]])
    # 降序排列
    idxes = np.argsort(-scores)
    classes = classes[idxes][:top_k]
    scores = scores[idxes][:top_k]
    bboxes = bboxes[idxes][:top_k]
    return classes, scores, bboxes


def bboxes_iou(bboxes1, bboxes2):
    """Computing iou between bboxes1 and bboxes2.
    Note: bboxes1 and bboxes2 can be multi-dimensional, but should broacastable.
    """
    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)
    # Intersection bbox and volume.
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    # Union volume.
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])
    iou = int_vol / (vol1 + vol2 - int_vol)
    return iou


# 非最大值抑制
def bboxes_nms(classes, scores, bboxes, nms_threshold=0.5):
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            # 计算两个box的IOU值
            overlap = bboxes_iou(bboxes[i], bboxes[(i+1):])
            # 逻辑或，小于门限值或者类别不相同就保留这个框
            keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i+1):] != classes[i])
            # 选取下一个
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)
    # np.where返回索引值
    idxes = np.where(keep_bboxes)
    return classes[idxes], scores[idxes], bboxes[idxes]






