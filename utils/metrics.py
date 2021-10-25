# -*- coding: utf-8 -*-
"""
@Project : 
@FileName: 
@Author  :penghr 
@Time    :202x/xx/xx xx:xx
@Desc  : 
"""
import math

import cv2
import numpy as np
import scipy.spatial
import torch
import torch.nn as nn
import torch.nn.functional as F


def LMDS_counting(fmap, img_name, f_loc):
    input_max = torch.max(fmap).item()

    ''' find local maxima'''
    keep = nn.functional.max_pool2d(fmap, (3, 3), stride=1, padding=1)
    keep = (keep == fmap).float()
    fmap = keep * fmap

    '''set the pixel valur of local maxima as 1 for counting'''
    fmap[fmap < 100.0 / 255.0 * input_max] = 0
    fmap[fmap > 0] = 1

    ''' negative sample'''
    if input_max < 0.1:
        fmap = fmap * 0

    count = int(torch.sum(fmap).item())

    kpoint = fmap.data.squeeze(0).squeeze(0).cpu().numpy()

    f_loc.write('{} {} '.format(img_name, count))
    return count, kpoint, f_loc


def generate_point_map(kpoint, f_loc, rate=1):
    '''obtain the location coordinates'''
    pred_coord = np.nonzero(kpoint)

    point_map = np.zeros((int(kpoint.shape[0] * rate), int(kpoint.shape[1] * rate), 3), dtype="uint8") + 255  # 22
    # count = len(pred_coor[0])
    coord_list = []
    for i in range(0, len(pred_coord[0])):
        h = int(pred_coord[0][i] * rate)
        w = int(pred_coord[1][i] * rate)
        coord_list.append([w, h])
        cv2.circle(point_map, (w, h), 2, (0, 0, 0), -1)

    for data in coord_list:
        f_loc.write('{} {} '.format(math.floor(data[0]), math.floor(data[1])))
    f_loc.write('\n')

    return point_map


def generate_bounding_boxes(kpoint, fname, resize):
    '''change the data path'''
    Img_data = cv2.imread(fname)
    # ori_Img_data = Img_data.copy()
    Img_data = cv2.resize(Img_data, resize)

    '''generate sigma'''
    pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    distances, locations = tree.query(pts, k=4)
    for index, pt in enumerate(pts):
        pt2d = np.zeros(kpoint.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if np.sum(kpoint) > 1:
            sigma = (distances[index][1] + distances[index][2] + distances[index][3]) * 0.1
        else:
            sigma = np.average(np.array(kpoint.shape)) / 2. / 2.  # case: 1 point
        sigma = min(sigma, min(Img_data.shape[0], Img_data.shape[1]) * 0.05)

        if sigma < 6:
            t = 2
        else:
            t = 2
        Img_data = cv2.rectangle(Img_data, (
            int((pt[0] * Img_data.shape[1] / resize[0] - sigma)), int((pt[1] * Img_data.shape[0] / resize[1] - sigma))),
                                 (int((pt[0] * Img_data.shape[1] / resize[0] + sigma)),
                                  int((pt[1] * Img_data.shape[0] / resize[1] + sigma))), (0, 255, 0), t)

    return Img_data
