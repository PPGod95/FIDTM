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
import warnings

import torch
import torch.nn as nn
import torch.utils.data

from test import test
from utils.data import *
from utils.model import *
from utils.datasets import *
from utils.general import *
from Networks.HR_Net_modify.seg_hrnet import get_seg_model

warnings.filterwarnings('ignore')


def train(pre_data, model, criterion, optimizer, epoch, args):
    train_loader = torch.utils.data.DataLoader(
        listDataset(pre_data,
                    transform=transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225]), ]),
                    task='train',
                    args=args),
        num_workers=args.workers,
        pin_memory=False,
        batch_size=args.batch_size,
        drop_last=False)

    args.lr = optimizer.param_groups[0]['lr']

    model.train()

    logger.info(('%10s' * 4) % ('Epoch', 'Samples', 'LRate', 'Loss'))
    pbar = enumerate(train_loader)
    pbar = tqdm(pbar, total=len(train_loader))
    for i, (fname, img, fidt_map, kpoint) in pbar:
        img = img.cuda()
        fidt_map = fidt_map.type(torch.FloatTensor).unsqueeze(1).cuda()
        d6 = model(img)

        if d6.shape != fidt_map.shape:
            print("the shape is wrong, please check. Both of prediction and GT should be [B, C, H, W].")
            exit()
        loss = criterion(d6, fidt_map)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        s = ('%10s' * 4) % (
        f'{epoch}/{args.epochs - 1}', epoch * len(train_loader.dataset), args.lr, round(float(loss)))
        pbar.set_description(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FIDTM')
    parser.add_argument('--gpu_id', type=str, default='1', help='gpu id')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--workers', type=int, default=8, help='load data workers')
    parser.add_argument('--dataset_path', type=str, default='dataset/NWPU', help='choice train dataset')
    parser.add_argument('--project', default='run/train', help='save results to project/name')
    parser.add_argument('--name', type=str, default='exp', help='save checkpoint directory')
    parser.add_argument('--pre_trained', type=str, default=None, help='pre-trained model directory')
    # parser.add_argument('--pre_trained', type=str, default='model/NWPU-Crowd/model_best_nwpu.pth',help='pre-trained model directory')

    parser.add_argument('--test_freq', type=int, default=30, help='print frequency')
    parser.add_argument('--crop_size', type=int, default=256, help='crop size for training')
    parser.add_argument('--epochs', type=int, default=1000, help='end epoch for training')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch for training')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5 * 1e-4, help='weight decay')
    parser.add_argument('--resize', type=tuple, default=(1440, 810), help='resize for input img')

    parser.add_argument('--best_pred', type=int, default=1e3, help='best pred')
    '''video demo'''
    parser.add_argument('--video_path', type=str, default=None, help='input video path ')
    args = parser.parse_args()

    '''fixed random seed '''
    setup_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    logging.basicConfig(format='%(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(args)

    # 加载数据
    dataset_path = args.dataset_path
    train_img_list = os.path.join(dataset_path, 'train/images')
    val_img_list = os.path.join(dataset_path, 'test/images')

    train_list = []
    for file_name in os.listdir(train_img_list):
        train_list.append(os.path.join(train_img_list, file_name))

    val_list = []
    for file_name in os.listdir(val_img_list):
        val_list.append(os.path.join(val_img_list, file_name))

    torch.set_num_threads(4)
    train_data = pre_data(train_list, args, train=True)
    val_data = pre_data(val_list, args, train=False)
    logger.info(f'train_size:{len(train_list)}, test_size:{len(val_list)}')

    # 加载模型
    model = get_seg_model(train=True)
    # model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': args.lr}], lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss(size_average=False).cuda()
    # criterion = nn.MSELoss(reduction='sum').cuda()

    save_path = os.path.join(args.project, args.name)
    if os.path.exists(save_path):
        save_path = save_path + str(len(os.listdir(args.project)))
        os.makedirs(save_path)
    else:
        os.makedirs(save_path)
    logger.info(f'file save to {save_path}')

    # 预训练模型
    if args.pre_trained:
        if os.path.isfile(args.pre_trained):
            print("=> loading checkpoint '{}'".format(args.pre_trained))
            checkpoint = torch.load(args.pre_trained)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args.start_epoch = checkpoint['epoch']
            args.best_pred = checkpoint['best_precision']
            print('pretrained model:', args.pre_trained)
        else:
            print("=> no checkpoint found at '{}'".format(args.pre_trained))

    # for name, param in model.named_parameters():  # 查看可优化的参数有哪些
    #     if param.requires_grad:
    #         print(name)
    best_pred = args.best_pred
    start_time = time.time()
    # begin training
    for epoch in range(args.start_epoch, args.epochs):
        train(train_data, model, criterion, optimizer, epoch, args)
        if epoch % args.test_freq == 0:
            # val
            precision = test(val_data, model, save_path, args)
            is_best = precision < best_pred
            best_pred = min(precision, best_pred)
            logger.info(f'* best MAE: {best_pred}, save_path: {save_path}')
            state = {'epoch': epoch + 1,
                    'pre_trained': args.pre_trained,
                    'state_dict': model.state_dict(),
                    'best_precision': best_pred,
                    'optimizer': optimizer.state_dict()}
            torch.save(state, os.path.join(save_path,'checkpoint.pth'))
            if is_best:
                torch.save(model, os.path.join(save_path, 'best.pt'))
        torch.save(model, os.path.join(save_path, 'last.pt'))
        # end training
    precision = test(val_data, model, save_path, args)

    logger.info(f'Finish training in {round((time.time()-start_time)/3600,2)}h \n * best MAE: {round(best_pred,2)}, results save to: {save_path}')
