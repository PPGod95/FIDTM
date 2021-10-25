from __future__ import division

import os.path
import warnings

from torchvision import datasets, transforms

import dataset

from config import return_args, args

from utils.data import *
from utils.model import *
from utils.metrics import *
from utils.general import *
from Networks.HR_Net.seg_hrnet import get_seg_model

warnings.filterwarnings('ignore')

# setup_seed(args.seed)

def validate(pre_data, model, save_path, args):
    print('begin test')
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(
                    dataset.listDataset(pre_data,
                                        shuffle=False,
                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                           std=[0.229, 0.224, 0.225]),]),
                                        args=args, train=False),
                                        batch_size=batch_size)

    model.eval()

    mae = 0.0
    mse = 0.0
    visi = []
    # index = 0
    loc_file = open(os.path.join(save_path, 'localization.txt'), 'w+')
    # if not os.path.exists('./local_eval/point_files'):
    #     os.makedirs('./local_eval/point_files')

    '''output coordinates'''
    # f_loc = open("./local_eval/point_files/A_localization.txt", "w+")

    for i, (fname, img, fidt_map, kpoint) in enumerate(test_loader):
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
            count, pred_kpoint, loc_file = LMDS_counting(d6, fname, loc_file)
            # # 画点图
            # point_map = generate_point_map(pred_kpoint, loc_file)
            # # 画框图
            # box_img = generate_bounding_boxes(pred_kpoint, os.path.join(args.dataset_path, 'test/images', fname[0]), args.resize)
            # # 热力图
            # show_fidt = show_fidt_map(d6.data.cpu().numpy())
            # gt_show = show_fidt_map(fidt_map.data.cpu().numpy())
            # print(type(point_map), type(box_img), type(show_fidt), type(gt_show))

            #     if not os.path.exists(args['save_path'] + '_box/'):
            #         os.makedirs(args['save_path'] + '_box/')
            # ori_img, box_img = generate_bounding_boxes(pred_kpoint, fname, args.resize)

            # res = np.hstack((img, gt_show, show_fidt, point_map, box_img))
            # cv2.imwrite(args.save_path + '_box/' + fname[0], res)

        gt_count = torch.sum(kpoint).item()
        mae += abs(gt_count - count)
        mse += abs(gt_count - count) * abs(gt_count - count)

        if i % 1 == 0:
            # print('{fname} Gt {gt:.2f} Pred {pred}'.format(fname=fname[0], gt=gt_count, pred=count))
            visi.append([img.data.cpu().numpy(), d6.data.cpu().numpy(), fidt_map.data.cpu().numpy(), fname])
            # index += 1

    mae = mae * 1.0 / (len(test_loader) * batch_size)
    mse = math.sqrt(mse / (len(test_loader)) * batch_size)

    print(' \n* MAE {mae:.3f}'.format(mae=mae), '* MSE {mse:.3f}'.format(mse=mse))

    for j in range(len(visi)):
        img = visi[j][0]
        output = visi[j][1]
        target = visi[j][2]
        fname = visi[j][3]
        save_results(img, target, output, save_path, fname[0])

    return mae


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    model = get_seg_model()
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec']
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    torch.set_num_threads(args.workers)
    print(args.best_pred, args.start_epoch)

    test_data = os.listdir('data/test')
    test_data.sort()

    for i in range(len(test_data)):
        test_data[i] = 'data/test/' + test_data[i]

    '''inference '''
    prec1, visi = validate(test_data, model, args)

    print('\nThe visualizations are provided in ', args['save_path'])

