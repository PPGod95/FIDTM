
import os
import logging
import argparse
import warnings

import torch.utils.data
from torchvision import datasets, transforms

from utils.data import *
from utils.model import *
from utils.datasets import *
from utils.metrics import *
from utils.general import *
from Networks.HR_Net.seg_hrnet import get_seg_model

warnings.filterwarnings('ignore')


def test(pre_data, model, save_path, args):
    print('begin test')
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(
        listDataset(pre_data,
                    transform=transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225]), ]),
                    task='test',
                    args=args),
        num_workers=args.workers,
        pin_memory=False,
        batch_size=batch_size)

    model.eval()

    mae, mse = 0.0, 0.0
    visi = []
    '''output coordinates'''
    loc_file = open(os.path.join(save_path, 'localization.txt'), 'w+')

    pbar = enumerate(test_loader)
    pbar = tqdm(pbar, total=len(test_loader))

    for i, (fname, img, fidt_map, kpoint) in pbar:
        img = img.cuda()

        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(fidt_map.shape) == 5:
            fidt_map = fidt_map.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        if len(fidt_map.shape) == 3:
            fidt_map = fidt_map.unsqueeze(0)

        with torch.no_grad():
            d6 = model(img)

            # 转换点图计数
            count, pred_kpoint, loc_file = LMDS_counting(d6, fname[0], loc_file)
            # 画点图
            point_map = generate_point_map(pred_kpoint, loc_file)

        gt_count = torch.sum(kpoint).item()
        mae += abs(gt_count - count)
        mse += abs(gt_count - count) * abs(gt_count - count)

        if i < 5:
            visi.append([img.data.cpu().numpy(), d6.data.cpu().numpy(), fidt_map.data.cpu().numpy(), fname[0]])

    mae = mae * 1.0 / (len(test_loader) * batch_size)
    mse = math.sqrt(mse / (len(test_loader)) * batch_size)

    print(' \n* MAE {mae:.3f}'.format(mae=mae), '* MSE {mse:.3f}'.format(mse=mse))

    for j in range(len(visi)):
        img = visi[j][0]
        output = visi[j][1]
        target = visi[j][2]
        img_name = visi[j][3]
        save_results(img, target, output, save_path, img_name)

    return mae


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FIDTM')
    parser.add_argument('--gpu_id', type=str, default='1', help='gpu id')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--workers', type=int, default=0, help='load data workers')
    parser.add_argument('--dataset_path', type=str, default='dataset/ShanghaiTech/part_A_final', help='choice train dataset')
    parser.add_argument('--project', default='run/test', help='save results to project/name')
    parser.add_argument('--name', type=str, default='exp', help='save checkpoint directory')
    parser.add_argument('--model', type=str, default='run/train/exp5/model_best.pth', help='pre-trained model directory')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size for test')
    # parser.add_argument('--resize', type=tuple, default=(1440, 810), help='resize for input img')

    args = parser.parse_args()

    # setup_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    logging.basicConfig(format='%(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(args)

    # 加载数据
    dataset_path = args.dataset_path
    test_img_list = os.path.join(dataset_path, 'test/images')
    test_list = []
    for file_name in os.listdir(test_img_list):
        test_list.append(os.path.join(test_img_list, file_name))

    torch.set_num_threads(4)
    test_data = pre_data(test_list, args, train=False)

    logger.info(f'test_size:{len(test_list)}')

    model = get_seg_model()
    # model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()

    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        print("=> no checkpoint found at '{}'".format(args.model))

    save_path = os.path.join(args.project, args.name)
    if os.path.exists(save_path):
        save_path = save_path + str(len(os.listdir(args.project)))
        os.makedirs(save_path)
    else:
        os.makedirs(save_path)
    logger.info(f'file save to {save_path}')

    precision = test(test_data, model, save_path, args)

    print('\nThe visualizations are provided in ', save_path)
