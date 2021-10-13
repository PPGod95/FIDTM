# -*- coding: utf-8 -*-
"""
@Project : 
@FileName: 
@Author  :penghr 
@Time    :202x/xx/xx xx:xx
@Desc  : 
"""


import numpy as np
import scipy.io as io
import os
root = './dataset/NWPU/images/'
train_list = []
eval_list = []
test_list = []


# for i in sorted(os.listdir(root)):
#     train_list.append(os.path.join(root,i))
# print(train_list)
arr = np.load('npydata/ShanghaiA_train.npy')
print(len(arr))
print(arr)
# mat = io.loadmat('/Users/hrpeng/Downloads/NWPU-Crowd/mats/0001.mat')
# print(mat['annBoxes'])

# mat = io.loadmat('/Users/hrpeng/Desktop/ShanghaiTech/part_A_final/train_data/ground_truth/GT_IMG_1.mat')
# a =mat['image_info'][0][0][0][0][0]
# print(mat.keys())
# print(mat['image_info'][0][0][0][0][0])