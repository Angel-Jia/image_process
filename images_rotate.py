import os
import json
import os.path as osp
from PIL import Image
import numpy as np
import math
import cv2

# 逆时针旋转角度为正，顺时针为负
angle = 0
angle = -30/180 * math.pi
rotate_matrix = [
    [math.cos(angle), math.sin(angle), 0],
    [-math.sin(angle), math.cos(angle), 0],
    [0, 0, 1]
]

trans_matrix = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]

zoom_matrix = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]

# 双线性插值
def bilinear_interpolation(img: np.ndarray, srcx: float, srcy: float, dst_width, dst_height):
    img_shape = img.shape
    
    srcx_int = int(srcx)
    srcy_int = int(srcy)

    if srcx_int < 0 or srcy_int < 0 or \
       srcx_int >= img_shape[1] - 1 or srcy >= img_shape[0] - 1:
        return np.array([0, 0, 0], dtype=np.uint8)

    x_diff = srcx - srcx_int
    y_diff = srcy - srcy_int

    v11 = img[srcy_int, srcx_int, :].astype(np.float32)
    v12 = img[srcy_int, srcx_int + 1, :].astype(np.float32)
    v21 = img[srcy_int + 1, srcx_int, :].astype(np.float32)
    v22 = img[srcy_int + 1, srcx_int + 1, :].astype(np.float32)

    # 双线性插值公式
    v = (1 - x_diff) * (1 - y_diff) * v11 + \
        x_diff * (1 - y_diff) * v12 + \
        (1 - x_diff) * y_diff * v21 + \
        x_diff * y_diff * v22
    
    return np.round(v).astype(np.uint8)

# 计算旋转后的图像的外接矩形
def cal_new_loc(img, matrix: np.ndarray):
    img_shape = img.shape
    loc00 = np.array([0, 0, 1], dtype=np.float32)
    loc01 = np.array([img_shape[1], 0, 1], dtype=np.float32)
    loc10 = np.array([0, img_shape[0], 1], dtype=np.float32)
    loc11 = np.array([img_shape[1], img_shape[0], 1], dtype=np.float32)

    new_loc00 = np.matmul(matrix, loc00.T).T
    new_loc01 = np.matmul(matrix, loc01.T).T
    new_loc10 = np.matmul(matrix, loc10.T).T
    new_loc11 = np.matmul(matrix, loc11.T).T
    new_loc = np.stack([new_loc00, new_loc01, new_loc10, new_loc11])
    print(new_loc)

    min_v = new_loc.min(axis=0)
    x_min, y_min = min_v[0], min_v[1]
    max_v = new_loc.max(axis=0)
    x_max, y_max = max_v[0], max_v[1]
    print(x_min, y_min, x_max, y_max)
    return x_min, y_min, x_max, y_max

def cal_new_img(new_img: np.ndarray, matrix: np.ndarray, img: np.ndarray, x_min, y_min, x_max, y_max):
    matrix_inv = np.linalg.inv(matrix)
    x_diff, y_diff = x_min, y_min
    new_img_shape = new_img.shape
    for i in range(new_img_shape[0]):
        for j in range(new_img_shape[1]):
            # 将new_img的原点坐标[0, 0]移动到旋转后的dst图像坐标[x_min, y_min]上
            dsty = i + y_diff
            dstx = j + x_diff
            # 根据dst坐标求src坐标，0.5是为了像素中心对齐
            src_ret = np.matmul(matrix_inv, np.array([dstx + 0.5, dsty + 0.5, 1], dtype=np.float32).T).T
            srcx = src_ret[0] - 0.5
            srcy = src_ret[1] - 0.5
            v = bilinear_interpolation(img, srcx, srcy, new_img_shape[1], new_img_shape[0])
            new_img[i, j, :] = v


if __name__ == '__main__':
    img = Image.open('resource/111.jpg')
    img = np.array(img)
    # img = np.zeros((200, 100, 3), np.uint8)

    ret = cal_new_loc(img, rotate_matrix)
    x_min, y_min, x_max, y_max = [int(round(x)) for x in ret]
    print(x_min, y_min, x_max, y_max)

    new_img = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
    cal_new_img(new_img, rotate_matrix, img, x_min, y_min, x_max, y_max)
    new_img = Image.fromarray(new_img)
    new_img.save('resource/222.jpg')

