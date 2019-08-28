import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torch.utils.data
import time
from dataloader import KITTIloader2015 as ls
from dataloader import KITTILoader as DA
import utils.logger as logger
from utils.utils import GERF_loss, smooth_L1_loss
from models.StereoNet8Xmulti import StereoNet
from os.path import join
import numpy as np

parser = argparse.ArgumentParser(description='StereoNet with Flyings3d')
parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[1.0, 1.0, 1.0, 1.0, 1.0])
parser.add_argument('--datapath', default='../dataset/kitti_2015/training/', help='datapath')
parser.add_argument('--epoch', type=int, default=300, help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=1,
                    help='batch size for training(default: 1)')
parser.add_argument('--itersize', default=1, type=int,
                    metavar='IS', help='iter size')
parser.add_argument('--test_bsize', type=int, default=1,
                    help='batch size for test(default: 1)')
parser.add_argument('--save_path', type=str, default='results/Kitti_2015/',
                    help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default='results/FlyingThings3D/checkpoint_14.pth', help='resume path')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=1, type=int,
                    metavar='SS', help='learning rate step size')
parser.add_argument('--gamma', '--gm', default=0.1, type=float,
                    help='learning rate decay parameter: Gamma')
parser.add_argument('--print_freq', type=int, default=10, help='print frequence')
parser.add_argument('--stages', type=int, default=3, help='the stage num of refinement')
parser.add_argument('--gpu', default='0', type=str, help='GPU ID')

args = parser.parse_args()


def main():
    global args
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(args.datapath)

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(train_left_img, train_right_img, train_left_disp, True),
        batch_size=args.train_bsize, shuffle=True, num_workers=1, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = logger.setup_logger(args.save_path + 'training.log')
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ':' + str(value))

    model = StereoNet(maxdisp=args.maxdisp)
    model = nn.DataParallel(model).cuda()
    model.apply(weights_init)

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=args.gamma)

    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    args.start_epoch = 0

    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format((args.resume)))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            log.info("=> loaded checkpoint '{}' (epeoch {})".format(args.resume, checkpoint['epoch']))
        else:
            log.info("=> no checkpoint found at '{}'".format(args.resume))
            log.info("=> will start from scratch.")
    else:
        log.info("Not Resume")

    min_erro = 100000
    max_epo = 0
    start_full_time = time.time()
    for epoch in range(args.start_epoch, args.epoch):
        log.info('This is {}-th epoch'.format(epoch))

        train(TrainImgLoader, model, optimizer, log, epoch)
        scheduler.step()

        erro = test(TestImgLoader, model, log)
        if erro < min_erro:
            max_epo = epoch
            min_erro = erro
            savefilename = args.save_path + 'finetune_checkpoint_{}.pth'.format(max_epo)
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict()},
                savefilename)
        log.info('MIN epoch %d total test erro = %.3f' % (max_epo, min_erro))
    log.info('full training time = {: 2f} Hours'.format((time.time() - start_full_time) / 3600))


def train(dataloader, model, optimizer, log, epoch=0):
    stages = args.stages
    losses = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)
    counter = 0

    model.train()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):

        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        outputs = model(imgL, imgR)  # 4 [1, 512, 768]

        loss = []
        for i in range(len(outputs)):
            loss.append(GERF_loss(disp_L, outputs[i]))

        counter += 1
        loss_all = sum(loss) / (args.itersize)
        loss_all.backward()
        if counter == args.itersize:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0

        for idx in range(stages):
            losses[idx].update(loss[idx].item() / args.loss_weights[idx])

        if batch_idx % args.print_freq == 0:
            info_str = ['Stage {} = {:.3f}({:.3f})'.format(x, losses[x].val, losses[x].avg) for x in range(stages)]

            info_str = '\t'.join(info_str)

            log.info('Epoch{} [{}/{}] {}'.format(
                epoch, batch_idx, length_loader, info_str))

            # vis
            # _, H, W = outputs[0].shape
            # all_results = torch.zeros((len(outputs) + 1, 1, H, W))
            # for j in range(len(outputs)):
            #     all_results[j, 0, :, :] = outputs[j][0, :, :] / 255.0
            # all_results[-1, 0, :, :] = disp_L[:, :] / 255.0
            # torchvision.utils.save_image(all_results, join(args.save_path + 'train_disp', "iter-%d.jpg" % batch_idx))
            # print(imgL)
            # im = np.array(imgL.cpu()[0, :, :, :].permute(1, 2, 0) * 255, dtype=np.uint8)
            #
            # cv.imwrite(join(args.save_path, "itercolor-%d.jpg" % batch_idx), im)

    info_str = '\t'.join(['Stage {} = {:.3f}'.format(x, losses[x].avg) for x in range(stages)])
    log.info('Average train loss = ' + info_str)


def test(dataloader, model, log):
    stages = args.stages
    erro_rate = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.eval()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()

        with torch.no_grad():
            outputs = model(imgL, imgR)

        for x in range(stages):
            output = outputs[x].cpu()

            res_disp = torch.zeros_like(disp_L)
            index = np.argwhere(disp_L > 0)
            res_disp[index[0][:], index[1][:], index[2][:]] = np.abs(disp_L[index[0][:], index[1][:], index[2][:]] - output[index[0][:], index[1][:], index[2][:]])
            erro = (res_disp[index[0][:], index[1][:], index[2][:]] < 3) | (res_disp[index[0][:], index[1][:], index[2][:]] < disp_L[index[0][:], index[1][:], index[2][:]] * 0.05)
            torch.cuda.empty_cache()
            erro_rate[x].update(1 - (float(torch.sum(erro)) / float(len(index[0]))))

        info_str = '\t'.join(['Stage {} = {:.3f}({:.3f})'.format(x, erro_rate[x].val * 100, erro_rate[x].avg * 100) for x in range(stages)])
        log.info('[{}/{}] {}'.format(batch_idx, length_loader, info_str))

        # _, H, W = outputs[0].shape
        # all_results = torch.zeros((len(outputs) + 1, 1, H, W))
        # for j in range(len(outputs)):
        #     all_results[j, 0, :, :] = outputs[j][0, :, :] / 255.0
        # all_results[-1, 0, :, :] = disp_L[:, :] / 255.0
        # torchvision.utils.save_image(all_results, join(args.save_path + 'test_disp', "iter-%d.jpg" % batch_idx))

    info_str = ', '.join(['Stage {}={:.3f}'.format(x, erro_rate[x].avg * 100) for x in range(stages)])
    log.info('Average test accuracy = ' + info_str)
    return erro_rate[2].avg * 100


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()
    if isinstance(m, nn.Conv3d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()


class AverageMeter(object):
    """Compute and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()


