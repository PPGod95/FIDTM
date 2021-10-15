# -*- coding: utf-8 -*-
"""
@Project : FIDTM
@FileName: train.py
@Author  :penghr 
@Time    :2021/10/15 10:00
@Desc  : 训练脚本
"""

import os
import time
import logging
import argparse

from config import args

# warnings.filterwarnings('ignore')
'''fixed random seed '''
# setup_seed(args.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info(args)

    dataset_path = args.dataset_path
    train_img_list = os.path.join(dataset_path, 'train/images')
    test_img_list = os.path.join(dataset_path, 'test/images')

    train_list = []
    for file_name in os.listdir(train_img_list):
        train_list.append(os.path.join(train_img_list, file_name))

    test_list = []
    for file_name in os.listdir(test_img_list):
        test_list.append(os.path.join(test_img_list, file_name))

    logger.info(f'train_size:{len(train_list)}, test_size:{len(test_list)}')
