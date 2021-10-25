# -*- coding: utf-8 -*-
"""
@Project : 
@FileName: 
@Author  :penghr 
@Time    :202x/xx/xx xx:xx
@Desc  : 
"""
import random

import cv2
import torch
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 输入固定情况下用true


def show_fidt_map(input):
    input[input < 0] = 0
    input = input[0][0]
    fidt_map = input
    fidt_map = fidt_map / np.max(fidt_map) * 255
    fidt_map = fidt_map.astype(np.uint8)
    fidt_map = cv2.applyColorMap(fidt_map, 2)
    return fidt_map
