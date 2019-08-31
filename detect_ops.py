"""
Detection ops for Yolov2
"""

import tensorflow as tf
import numpy as np


# decode from the detection feature
# 解码函数：输入：detection_feat:darknet输出的结果， feat_size：darknet输出的结果尺寸
#               num_classes:训练中的总类别个数， anchors：先验框，共设置了5个
def decode(detection_feat, feat_sizes=(13, 13), num_classes=80,
           anchors=None):
    # 13*13
    H, W = feat_sizes
    # 设置了5个
    num_anchors = len(anchors)
    # num_classes+5表示预测80和类别和4个坐标值和1个置信度
    # reshape 为[batch_size, 13*13, 5，85], -1是用其他其他参数确定这个维度的值为多少
    detection_results = tf.reshape(detection_feat, [-1, H * W, num_anchors,
                                                   num_classes + 5])
    # x, y的坐标值
    bbox_xy = tf.nn.sigmoid(detection_results[:, :, :, 0:2])
    # w, h偏移值
    bbox_wh = tf.exp(detection_results[:, :, :, 2:4])
    # 置信度
    obj_probs = tf.nn.sigmoid(detection_results[:, :, :, 4])
    # 类别
    class_probs = tf.nn.softmax(detection_results[:, :, :, 5:])
    # constant常量值，不会被改变
    anchors = tf.constant(anchors, dtype=tf.float32)
    # 生成range数 如0:1
    height_ind = tf.range(H, dtype=tf.float32)
    width_ind = tf.range(W, dtype=tf.float32)
    # tf.meshgrid: 在平面上画格子
    x_offset, y_offset = tf.meshgrid(height_ind, width_ind)
    # reshape为[1, -1 ,1],有几个格子，就会有几个x/y值，因此将中间的设置为-1
    x_offset = tf.reshape(x_offset, [1, -1, 1])
    y_offset = tf.reshape(y_offset, [1, -1, 1])

    # decode
    bbox_x = (bbox_xy[:, :, :, 0] + x_offset) / W
    bbox_y = (bbox_xy[:, :, :, 1] + y_offset) / H
    # anchors[:,0]对应5个先验框的第一个参数，anchor[:,1]对应先验框的第二个参数
    bbox_w = bbox_wh[:, :, :, 0] * anchors[:, 0] / W * 0.5
    bbox_h = bbox_wh[:, :, :, 1] * anchors[:, 1] / H * 0.5
    # 拼接坐标的四个值
    bboxes = tf.stack([bbox_x - bbox_w, bbox_y - bbox_h,
                       bbox_x + bbox_w, bbox_y + bbox_h], axis=3)
    # print(bboxes)

    return bboxes, obj_probs, class_probs
