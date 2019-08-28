# ------------------------------------------------------------------------------
# Copyright (c) NKU
# Licensed under the MIT License.
# Written by Xuanyi Li (xuanyili.edu@gmail.com)
# ------------------------------------------------------------------------------
import os
import torch
import torch.nn.functional as F
# import cv2 as cv
import numpy as np
def GERF_loss(GT, pred):
    # mask = (GT < args.maxdisp) & (GT >= 0)
    mask = GT > 0 
    mask.detach_()
    # print(mask.size(), GT.size(), pred.size())
    count = len(torch.nonzero(mask))
    # print(count)
    if count == 0:
        count = 1
    return torch.sum(torch.sqrt(torch.pow(GT[mask] - pred[mask], 2) + 4) /2 - 1) / count

def smooth_L1_loss(GT, pred, args):

    mask = GT < args.maxdisp
    mask.detach_()
    # loss = F.smooth_l1_loss(pred[mask], GT[mask], size_average=True)
    loss = (pred[mask] - GT[mask]).abs().mean()
    return loss


