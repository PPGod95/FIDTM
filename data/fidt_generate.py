# -*- coding: utf-8 -*-
"""
@Project : 
@FileName: 
@Author  :penghr 
@Time    :2021/11/xx xx:xx
@Desc  : FIDTM-train/dataset/FIDTM/
                                ├── test
                                │   ├── gt_fidt_map
                                │   │   └── IMG_8.h5
                                │   ├── gt_show
                                │   │   └── IMG_8.jpg
                                │   ├── images
                                │   │   └── IMG_8.jpg
                                │   └── labels
                                │       └── IMG_8.txt
                                └── train
                                    ├── gt_fidt_map
                                    │   └── IMG_1.h5
                                    ├── gt_show
                                    │   └── IMG_1.jpg
                                    ├── images
                                    │   └── IMG_1.jpg
                                    └── labels
                                        └── IMG_1.txt
原始数据集分为train&test，各目录下有images和labels文件夹，运行脚本生成gt_show以及gt_fidt_map文件夹，其中gt_show为可视化标注不参与训练，gt_fidt_map为生成的fidtmap和kpoint字典，参与下一步训练。
"""

import math
import os

import cv2
import h5py
import torch
import numpy as np
from tqdm import tqdm

# 生成路径
dataset_path = '../dataset/FIDTM'
label_type = 'txt'
train_path = os.path.join(dataset_path, 'train')
test_path = os.path.join(dataset_path, 'test')

train_img_path = os.path.join(train_path, 'images')
test_img_path = os.path.join(test_path, 'images')

train_label_path = os.path.join(train_path, 'labels')
test_label_path = os.path.join(test_path, 'labels')

train_gt_map = train_img_path.replace('images', 'gt_fidt_map')
test_gt_map = test_img_path.replace('images', 'gt_fidt_map')

train_gt_show = train_img_path.replace('images', 'gt_show')
test_gt_show = test_img_path.replace('images', 'gt_show')

path_list = [train_gt_map, test_gt_map, train_gt_show, test_gt_show]

for i in path_list:
    os.makedirs(i, exist_ok=True)

train_list = []
for fs in os.listdir(train_img_path):
    train_list.append(os.path.join(train_img_path, fs))

test_list = []
for fs in os.listdir(test_img_path):
    test_list.append(os.path.join(test_img_path, fs))

img_paths = train_list + test_list
img_paths.sort()


def fidt_generate(im_data, gt_data, lamda):
    size = im_data.shape
    new_im_data = cv2.resize(im_data, (lamda * size[1], lamda * size[0]), 0)

    new_size = new_im_data.shape
    d_map = (np.zeros([new_size[0], new_size[1]]) + 255).astype(np.uint8)
    gt_data = lamda * gt_data

    for o in range(0, len(gt_data)):
        x = np.max([1, math.floor(gt_data[o][1])])
        y = np.max([1, math.floor(gt_data[o][0])])
        if x >= new_size[0] or y >= new_size[1]:
            continue
        d_map[x][y] = d_map[x][y] - 255

    distance_map = cv2.distanceTransform(d_map, cv2.DIST_L2, 0)
    distance_map = torch.from_numpy(distance_map)
    distance_map = 1 / (1 + torch.pow(distance_map, 0.02 * distance_map + 0.75))
    distance_map = distance_map.numpy()
    distance_map[distance_map < 1e-2] = 0

    return distance_map

print('开始生成训练数据')

with tqdm(total=len(img_paths)) as pbar:
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if label_type == 'txt':
            gt = np.loadtxt(img_path.replace('images', 'labels').replace('.jpg', '.txt'))[:, 0:2].round(8)
        elif label_type == 'npy':
            gt = np.load(img_path.replace('images', 'labels').replace('.jpg', '.npy')).round(8)
        elif label_type == 'mat':
            gt = np.loadtxt(img_path.replace('images', 'labels').replace('.jpg', '.mat'))[:, 0:2].round(8)
        '''最关键，根据标签生成fidt图'''
        fidt_map = fidt_generate(img, gt, 1)

        # cv2.imshow('1', fidt_map)
        # cv2.waitKey(0)

        '''标签对应像素为1其余为0'''
        kpoint = np.zeros((img.shape[0], img.shape[1]))
        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                kpoint[int(gt[i][1]), int(gt[i][0])] = 1

        # cv2.imshow('1', kpoint)
        # cv2.waitKey(0)

        '''保存成h5文件，其实就是字典，后期优化'''
        with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'gt_fidt_map'), 'w') as hf:
            hf['fidt_map'] = fidt_map
            hf['kpoint'] = kpoint
        pbar.update()

        '''可视化，可以不要'''
        try:
            fidt_map1 = fidt_map
            fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
            fidt_map1 = fidt_map1.astype(np.uint8)
            fidt_map1 = cv2.applyColorMap(fidt_map1, 2)
            cv2.imwrite(img_path.replace('images', 'gt_show'), fidt_map1)
        except Exception as e:
            print(img_path,e)

        # cv2.imshow('1', fidt_map1)
        # cv2.waitKey(0)

    print('完成')
