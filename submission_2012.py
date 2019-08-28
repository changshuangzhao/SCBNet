from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.utils.data
import time
from models.StereoNet8Xmulti import StereoNet
import skimage
from dataloader import KITTI_submission_loader2012 as DA
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--datapath', default='../dataset/kitti_2012/testing/',
                    help='select model')
parser.add_argument('--loadmodel', default='./results/Kitti_2012/finetune_checkpoint_295.pth',
                    help='loading model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


test_left_img, test_right_img = DA.dataloader(args.datapath)
model = StereoNet(3, 3, args.maxdisp)
model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


def test(imgL,imgR):
    model.eval()

    if args.cuda:
        imgL = torch.FloatTensor(imgL).cuda()
        imgR = torch.FloatTensor(imgR).cuda()

    with torch.no_grad():
        output = model(imgL, imgR)
    pred_disp = torch.squeeze(output[2]).cpu().numpy()
    return pred_disp


def main():
    imagenet_stats = {'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]}
    processed = transforms.Compose([transforms.ToTensor(), transforms.Normalize(**imagenet_stats)])

    for inx in range(len(test_left_img)):
        imgL_o = Image.open(test_left_img[inx]).convert('RGB')
        imgR_o = Image.open(test_right_img[inx]).convert('RGB')
        imgL = processed(imgL_o).numpy()
        imgR = processed(imgR_o).numpy()

        imgL = np.reshape(imgL, [1, 3, imgL.shape[1], imgL.shape[2]])
        imgR = np.reshape(imgR, [1, 3, imgR.shape[1], imgR.shape[2]])

        # pad to (384, 1248)
        top_pad = 384 - imgL.shape[2]
        left_pad = 1248 - imgL.shape[3]
        imgL = np.lib.pad(imgL, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)
        imgR = np.lib.pad(imgR, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)

        start_time = time.time()
        pred_disp = test(imgL, imgR)
        print('time = %.2f' % (time.time() - start_time))

        top_pad = 384 - imgL_o.height
        left_pad = 1248 - imgL_o.width
        img = pred_disp[top_pad:, :-left_pad]
        skimage.io.imsave('./results/Kitti_2012/submission_disp/' + test_left_img[inx].split('/')[-1], (img * 256).astype('uint16'))


if __name__ == '__main__':
    main()







