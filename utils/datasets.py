# -*- coding: utf-8 -*-
"""
@Project : 
@FileName: 
@Author  :penghr 
@Time    :202x/xx/xx xx:xx
@Desc  : 
"""
import os
import random

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from torchvision import datasets, transforms

class listDataset(Dataset):
    def __init__(self, root, transform=None, task='train', args=None):
        if task == 'train':
            random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.task = task
        self.args = args

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        fname = self.lines[index]['fname']
        img = self.lines[index]['img']
        kpoint = self.lines[index]['kpoint']
        fidt_map = self.lines[index]['fidt_map']


        '''data augmention'''
        if self.task == 'train':
            if random.random() > 0.5:
                fidt_map = np.fliplr(fidt_map)
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                kpoint = np.fliplr(kpoint)


        fidt_map = fidt_map.copy()
        kpoint = kpoint.copy()
        img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        '''crop size'''
        if self.task == 'train':
            # fidt_map = torch.from_numpy(fidt_map).cuda()
            fidt_map = torch.from_numpy(fidt_map)
            width = self.args.crop_size
            height = self.args.crop_size
            crop_size_x = random.randint(0, img.shape[1] - width)
            crop_size_y = random.randint(0, img.shape[2] - height)
            img = img[:, crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height]
            kpoint = kpoint[crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height]
            fidt_map = fidt_map[crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height]

        '''需要resize使所有输入尺寸一样才能多batch_size'''
        # if self.task == 'test':
        #     # fidt_map = torch.from_numpy(fidt_map).cuda()
        #     fidt_map = torch.from_numpy(fidt_map)
        #     width = self.args.resize[0]
        #     height = self.args.resize[1]
        #     # crop_size_x = random.randint(0, img.shape[1] - width)
        #     # crop_size_y = random.randint(0, img.shape[2] - height)
        #
        #     img = img[:, 0:width, 0:height]
        #     kpoint = kpoint[0:width, 0:height]
        #     fidt_map = fidt_map[0:width, 0:height]
        #     print(img.shape)
        #     print(kpoint.shape)
        #     print(fidt_map.shape)



        return fname, img, fidt_map, kpoint

