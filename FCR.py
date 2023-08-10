from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import models
import time
import math

def FCR(bn_before,bn,im_sq,im_sq_before,percent):
    total= bn.shape[0]
    t = im_sq[int(0.01 * percent * total)]
    t_before = im_sq_before[int(0.01 * percent * total)]

    mask1 = bn_before.gt(t_before)
    mask2 = bn.gt(t)
    mask = mask2.int() - mask1.int()
    change = mask.eq(1)
    change_numbers = torch.sum(change)
    FCR = change_numbers / int(0.01 * percent * total)
    return FCR

def Golden_Ratio(bn_before,bn,im_sq,im_sq_before,start,end):
    middle=int((start+end)/2)
    FCR_ = FCR(bn_before, bn, im_sq, im_sq_before, middle)
    if(start+1==end):
        if (FCR_>=0.01):
            return start
        else:
            return end
    if (FCR_<=0.01):
        return Golden_Ratio(bn_before,bn,im_sq,im_sq_before,start,middle)
    else:
        return Golden_Ratio(bn_before,bn,im_sq,im_sq_before,middle,end)

if __name__=="__main__":
# Training settings
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='training dataset (default: cifar10)')
    parser.add_argument('--arch', default='vgg', type=str,
                        help='architecture to use')
    parser.add_argument('--depth', default=19, type=int,
                        help='depth of the neural network')

    parser.add_argument('--save', default=' ', type=str, metavar='PATH')
    parser.add_argument('--checkpoint', default=' ', type=str, metavar='PATH')
    parser.add_argument('--start_epoch', default=158, type=int)
    parser.add_argument('--end_epoch', default=159, type=int)
    args = parser.parse_args()

    args.save='FCR/{}_{}_FCR'.format(args.arch,args.dataset)
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if(args.arch=='vgg'):
        if(args.dataset=='cifar10'):
            args.checkpoint = './vgg-cifar10/sparsity'
        if (args.dataset == 'cifar100'):
            args.checkpoint = './vgg--cifar100/sparsity'
    if (args.arch == 'resnet'):
        if (args.dataset == 'cifar10'):
            args.checkpoint = './resnet-cifar10/sparsity'
        if (args.dataset == 'cifar100'):
            args.checkpoint = './resnet-cifar100/sparsity'
    if (args.arch == 'densenet'):
        if (args.dataset == 'cifar10'):
            args.checkpoint = './densnet-cifar10/sparsity'
        if (args.dataset == 'cifar100'):
            args.checkpoint = './densenet-cifar100/sparsity'

    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
    checkpoint = torch.load('{}/159epoch-L1.pth.tar'.format(args.checkpoint))
    model.load_state_dict(checkpoint['state_dict'])

    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]
    bn = torch.zeros(total)
    bn_before = torch.zeros(total)
    mask1 = torch.zeros(total)
    mask2 = torch.ones(total)

    for epoch in range(args.start_epoch,args.end_epoch):
        if epoch==0:
            continue
        checkpoint = torch.load('{}/{}epoch-L1.pth.tar'.format(args.checkpoint,epoch))
        checkpoint_before = torch.load('{}/{}epoch-L1.pth.tar'.format(args.checkpoint,epoch-1))

        model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
        model.load_state_dict(checkpoint['state_dict'])
        model_before = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
        model_before.load_state_dict(checkpoint_before['state_dict'])

        index = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.shape[0]
                bn[index:(index + size)] = m.weight.data.abs().clone()
                index += size
        index = 0
        for m in model_before.modules():
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.shape[0]
                bn_before[index:(index + size)] = m.weight.data.abs().clone()
                index += size

        im_sq, idx = torch.sort(bn)
        im_sq_before, idx_before = torch.sort(bn_before)

        golden_ratio=Golden_Ratio(bn_before,bn,im_sq,im_sq_before,1,99)
        print(golden_ratio)

#python FCR.py --dataset cifar10 --arch vgg --depth 19
#python FCR.py --dataset cifar10 --arch densenet --depth 40
#python FCR.py --dataset cifar10 --arch resnet --depth 164
#python FCR.py --dataset cifar100 --arch vgg --depth 19
#python FCR.py --dataset cifar100 --arch densenet --depth 40
#python FCR.py --dataset cifar100 --arch resnet --depth 164