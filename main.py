import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torch.utils.data
import time
from dataloader import listflowfile as lt 
from dataloader import SecenFlowLoader as DA
import utils.logger as logger
from utils.utils import GERF_loss, smooth_L1_loss
from models.StereoNet8Xmulti import StereoNet
from os.path import join

parser = argparse.ArgumentParser(description='StereoNet with Flyings3d')
parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--datapath', default='../dataset/', help='datapath')
parser.add_argument('--epoch', type=int, default=20, help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=1,
                    help='batch size for training(default: 1)')
parser.add_argument('--test_bsize', type=int, default=1,
                    help='batch size for test(default: 1)')
parser.add_argument('--save_path', type=str, default='results/FlyingThings3D/',
                    help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default=None, help='resume path')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=1, type=int, 
                    metavar='SS', help='learning rate step size')
parser.add_argument('--gamma', '--gm', default=0.7, type=float,
                    help='learning rate decay parameter: Gamma')
parser.add_argument('--print_freq', type=int, default=100, help='print frequence')
parser.add_argument('--stages', type=int, default=3, help='the stage num of refinement')
parser.add_argument('--gpu', default='0', type=str, help='GPU ID')

args = parser.parse_args()


def main():
    global args
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath)
    TrainImgLoader = torch.utils.data.DataLoader(DA.myImageFloder(train_left_img, train_right_img, train_left_disp, True), batch_size=args.train_bsize, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
    TestImgLoader = torch.utils.data.DataLoader(DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False), batch_size=args.test_bsize, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = logger.setup_logger(args.save_path + 'training.log')
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ':' + str(value))
    
    model = StereoNet(maxdisp=args.maxdisp)
    model = nn.DataParallel(model).cuda()
    model.apply(weights_init)

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format((args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}' ".format(args.resume))
        else:
            log.info("=> no checkpoint found at '{}'".format(args.resume))
            log.info("=> will start from scratch.")
    else:
        log.info("Not Resume")

    start_full_time = time.time()
    for epoch in range(args.start_epoch, args.epoch):
        log.info('This is {}-th epoch'.format(epoch))
        scheduler.step()

        train(TrainImgLoader, model, optimizer, log, epoch)

        savefilename = args.save_path + 'checkpoint_{}.pth'.format(epoch)
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, savefilename)

        test(TestImgLoader, model, log)
    log.info('full training time = {: 2f} Hours'.format((time.time() - start_full_time) / 3600))


def train(dataloader, model, optimizer, log, epoch=0):
    stages = args.stages
    losses = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.train()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        outputs = model(imgL, imgR)  # 4 [1, 512, 768]

        loss = []
        for i in range(len(outputs)):
            loss.append(GERF_loss(disp_L, outputs[i]))

        loss_all = sum(loss)
        loss_all.backward()
        optimizer.step()
        optimizer.zero_grad()

        for idx in range(stages):
            losses[idx].update(loss[idx].item())

        if batch_idx % args.print_freq == 0:
            info_str = ['Stage {} = {:.2f}({:.2f})'.format(x, losses[x].val, losses[x].avg) for x in range(stages)]

            info_str = '\t'.join(info_str)
            log.info('Epoch{} [{}/{}] {}'.format(epoch, batch_idx, length_loader, info_str))

            # vis
            _, H, W = outputs[0].shape
            all_results = torch.zeros((len(outputs) + 1, 1, H, W))
            for j in range(len(outputs)):
                all_results[j, 0, :, :] = outputs[j][0, :, :] / 255.0
            all_results[-1, 0, :, :] = disp_L[:, :] / 255.0
            torchvision.utils.save_image(all_results, join(args.save_path + 'train_disp', "iter-%d.jpg" % batch_idx))
            # print(imgL)
            # im = np.array(imgL.cpu()[0, :, :, :].permute(1, 2, 0) * 255, dtype=np.uint8)
            #
            # cv.imwrite(join(args.save_path, "itercolor-%d.jpg" % batch_idx), im)

    info_str = '\t'.join(['Stage {} = {:.2f}'.format(x, losses[x].avg) for x in range(stages)])
    log.info('Average train loss = ' + info_str)


def test(dataloader, model, log):

    stages = args.stages
    EPES = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.eval()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        mask = (disp_L < args.maxdisp) & (disp_L >= 0)
        
        with torch.no_grad():
            outputs = model(imgL, imgR)
            for x in range(stages):
                if len(disp_L[mask]) == 0:
                    EPES[x].update(0)
                    continue
                EPES[x].update((outputs[x][:, 4:, :][mask] - disp_L[mask]).abs().mean())

        if batch_idx % args.print_freq == 0:
            info_str = '\t'.join(['Stage {} = {:.2f}({:.2f})'.format(x, EPES[x].val, EPES[x].avg) for x in range(stages)])
            log.info('[{}/{}] {}'.format(batch_idx, length_loader, info_str))

            _, H, W = outputs[0][:, 4:, :].shape
            all_results = torch.zeros((len(outputs)+1, 1, H, W))
            for j in range(len(outputs)):
                all_results[j, 0, :, :] = outputs[j][:, 4:, :][0, :, :]/255.0
            all_results[-1, 0, :, :] = disp_L[:, :]/255.0
            torchvision.utils.save_image(all_results, join(args.save_path + 'test_disp', "iter-%d.jpg" % batch_idx))
            # print(imgL)
            # im = np.array(imgL[0,:,:,:].permute(1,2,0)*255, dtype=np.uint8)
            # print(im.shape)
            # cv.imwrite(join(args.save_path, "itercolor-%d.jpg" % batch_idx),im)

    info_str = ', '.join(['Stage {}={:.2f}'.format(x, EPES[x].avg) for x in range(stages)])
    log.info('Average test EPE = ' + info_str)


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
        self.val= 0
        self.avg= 0
        self.sum= 0
        self.count= 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()


