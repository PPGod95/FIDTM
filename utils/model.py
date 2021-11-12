# -*- coding: utf-8 -*-
"""
@Project : 
@FileName: 
@Author  :penghr 
@Time    :202x/xx/xx xx:xx
@Desc  : 
"""
import os

import cv2
import numpy as np


def save_results(input_img, gt_data, density_map, output_dir, fname):
    density_map[density_map < 0] = 0

    gt_data = 255 * gt_data / np.max(gt_data)
    gt_data = gt_data[0][0]
    gt_data = gt_data.astype(np.uint8)
    gt_data = cv2.applyColorMap(gt_data, 2)

    density_map = 255 * density_map / np.max(density_map)
    density_map = density_map[0][0]
    density_map = density_map.astype(np.uint8)
    density_map = cv2.applyColorMap(density_map, 2)

    result_img = np.hstack((gt_data, density_map))

    cv2.imwrite(os.path.join('.', output_dir, fname), result_img)
