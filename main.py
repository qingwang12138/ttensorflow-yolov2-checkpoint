"""
Demo for yolov2
"""

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

from model import darknet
from detect_ops import decode
from utils import preprocess_image, postprocess, draw_detection
from config import anchors, class_names


input_size = (416, 416)
image_file = "./test_data/car.jpg"
image = cv2.imread(image_file)
# 对于720*1280*3的图片 image.shape[0]=720, image.shape[1]=1280,image.shape[2]=3
image_shape = image.shape[:2]
# preprocess_image的主要工作：
# 缩放图片尺寸（缩放为[416,416]）并扩展batch_size维度
image_cp = preprocess_image(image, input_size)
"""
image = Image.open(image_file)
image_cp = image.resize(input_size, Image.BICUBIC)
image_cp = np.array(image_cp, dtype=np.float32)/255.0
image_cp = np.expand_dims(image_cp, 0)
#print(image_cp)
"""

# tf.placeholder：占位符,用于定义一个过程，在执行的时候再赋值，参数值为[data type,input_size,input_size, channel]
# input[0]=416, input[1]=416, 3是图像通道数
images = tf.placeholder(tf.float32, [1, input_size[0], input_size[1], 3])

# in model.py  yolov2的backbone网络 darknet
detection_feat = darknet(images)
# feat的尺寸 print(feat_sizes)的结果为(416/32, 416/32)=(13, 13)
feat_sizes = input_size[0] // 32, input_size[1] // 32
# decode的作用：解码预测框，返回类别预测和置信度值
detection_results = decode(detection_feat, feat_sizes, len(class_names), anchors)

checkpoint_path = "D:\\reference\\5-dataset\\yolo2_coco_weights\\yolo2_coco.ckpt"

# 保存训练中的权值
saver = tf.train.Saver()
with tf.Session() as sess:
    # saver.restore读出之前训练的权值
    saver.restore(sess, checkpoint_path)
    # feed_dict的作用是将images用image_cp替换掉
    bboxes, obj_probs, class_probs = sess.run(detection_results, feed_dict={images: image_cp})

# postprocess的作用：
# 计算IOU值, NMS操作
bboxes, scores, class_inds = postprocess(bboxes, obj_probs, class_probs,
                                         image_shape=image_shape)
# 画出检测框
img_detection = draw_detection(image, bboxes, scores, class_inds, class_names)

# 保存检测好的图像
cv2.imwrite("./detected_data/bird.jpg", img_detection)
# 显示检测好的图像
cv2.imshow("detection results", img_detection)
# 一直显示，知道按下任意键退出
cv2.waitKey(0)



