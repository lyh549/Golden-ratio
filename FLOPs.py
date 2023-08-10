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
from thop import profile

if __name__=="__main__":
# Training settings
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
    parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
    parser.add_argument('--pruned_model', default='', type=str, metavar='PATH',
                        help='path to the pruned model')
    parser.add_argument('--arch', default='resnet', type=str,
                        help='architecture to use')
    parser.add_argument('--depth', default=164, type=int,
                        help='depth of the neural network')

    args = parser.parse_args()

    if args.pruned_model:
        checkpoint = torch.load(args.pruned_model)
        model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
    
    num_parameters = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (num_parameters/1e6))
    input = torch.randn(1, 3, 32, 32)
    flops, params = profile(model,inputs=(input,))
    print("params: ",params)
    print("FLOPs: ",flops)
    #python FLOPs.py --resume resnet-baseline/0epoch-L1.pth.tar
    #python FLOPs.py --refine resnet-baseline/0epoch-L1.pth.tar
