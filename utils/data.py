# -*- coding: utf-8 -*-
"""
@Project : 
@FileName: 
@Author  :penghr 
@Time    :202x/xx/xx xx:xx
@Desc  : 
"""

import os

import h5py
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def pre_data(image_list, args, train):
    print("Pre_load dataset ......")
    data_keys = {}
    count = 0
    for j in tqdm(range(len(image_list))):
        Img_path = image_list[j]
        fname = os.path.basename(Img_path)
        img, fidt_map, kpoint = load_data_fidt(Img_path, args, train)

        if min(fidt_map.shape[0], fidt_map.shape[1]) < 256 and train == True:  # ignore some small resolution images
            continue
        blob = {}
        blob['img'] = img
        blob['kpoint'] = np.array(kpoint)
        blob['fidt_map'] = fidt_map
        blob['fname'] = fname
        data_keys[count] = blob
        count += 1

    return data_keys


def load_data_fidt(img_path, args, train=True):
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'gt_fidt_map')
    img = Image.open(img_path).convert('RGB')

    while True:
        try:
            gt_file = h5py.File(gt_path)
            kpoint = np.asarray(gt_file['kpoint'])
            fidt_map = np.asarray(gt_file['fidt_map'])
            break
        except OSError:
            print("path is wrong, can not load ", img_path)
            cv2.waitKey(1000)  # Wait a bit

    # img = img.copy()
    # fidt_map = fidt_map.copy()
    # kpoint = kpoint.copy()

    return img, fidt_map, kpoint
