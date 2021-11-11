# -*- coding: utf-8 -*-
"""
@Project : 
@FileName: 
@Author  :penghr 
@Time    :2021/10/19 10:30
@Desc  : 
"""

import os
import csv
import time
import argparse
import logging
import warnings

import cv2
import numpy as np
import torch.nn
from torchvision import transforms

from utils.metrics import *
from utils.general import *
from Networks.HR_Net.seg_hrnet import get_seg_model

warnings.filterwarnings('ignore')


def inference(model, source_list, save_path):
    model.eval()
    img_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor_transform = transforms.ToTensor()
    with torch.no_grad():
        start_time = time.time()
        loc_file = open(os.path.join(save_path, 'localization.txt'), 'w+')
        for img_name in source_list:
            img_path = os.path.join(args.source, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, args.resize)
            ori_img = img.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = tensor_transform(img)
            img = img_transform(img).unsqueeze(0)

            d6 = model(img)
            # 转换点图计数
            count, pred_kpoint, loc_file = LMDS_counting(d6, img_name, loc_file)
            # 画点图
            point_map = generate_point_map(pred_kpoint, loc_file)
            # 画框图
            box_img = generate_bounding_boxes(pred_kpoint, img_path, args.resize)
            # 热力图
            fidt_map = show_fidt_map(d6.data.cpu().numpy())

            # res = np.hstack((ori_img, fidt_map, point_map, box_img))
            # 堆叠图像
            res1 = np.hstack((ori_img, fidt_map))
            res2 = np.hstack((box_img, point_map))
            res = np.vstack((res1, res2))
            cv2.putText(res, "Count:" + str(count), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imwrite(save_path + '/' + img_name.split('.')[0] + '_' + str(count) + '.jpg', res)
            with open(save_path + '/results.csv', 'a') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(([img_name, str(count)]))
            print('{fname} Pred {pred}'.format(fname=img_name, pred=count))

        print('平均单张图片时间：', (time.time() - start_time) / len(source_list), 's')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FIDTM')
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--workers', type=int, default=16, help='load data workers')
    parser.add_argument('--source', type=str, default='dataset/FIDTM/test/images', help='choice inference dataset')
    parser.add_argument('--project', default='run/inference', help='save results to project/name')
    parser.add_argument('--name', type=str, default='exp', help='save checkpoint directory')
    parser.add_argument('--model', type=str, default='model/NWPU-Crowd/model_best_nwpu.pth', help='pre-trained model directory')
    parser.add_argument('--resize', type=tuple, default=(1440, 810), help='resize for input img')

    '''video demo'''
    parser.add_argument('--video_path', type=str, default=None, help='input video path ')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    logging.basicConfig(format='%(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(args)

    save_path = os.path.join(args.project, args.name)
    if os.path.exists(save_path):
        save_path = save_path + str(len(os.listdir(args.project)))
        os.makedirs(save_path)
    else:
        os.makedirs(save_path)
    logger.info(f'results save to {save_path}')

    source_path = args.source
    source_list = os.listdir(source_path)
    source_list.sort()

    model = get_seg_model()
    model = torch.nn.DataParallel(model, device_ids=[0])
    model = model.cuda()

    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        print("=> no checkpoint found at '{}'".format(args.model))

    inference(model, source_list, save_path)
