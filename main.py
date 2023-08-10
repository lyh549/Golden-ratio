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
np.set_printoptions(suppress=True)
torch.set_printoptions(threshold=np.inf)

# additional subgradient descent on the sparsity-induced penalty term
def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.to(device), target.to(device)
            #data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        loss.backward()
        if args.sr:
            updateBN()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
    FCR_print()

def FCR_print():
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]
    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size
    y, i = torch.sort(bn)
    global mask1,mask2,mask      
    for percent in range(1,10):
        t=y[int(0.1*percent*total)]
        mask1[percent-1]=bn.gt(t)
        mask[percent-1]=mask2[percent-1].int()-mask1[percent-1].int()
        mask2[percent-1]=mask1[percent-1]
        count=mask[percent-1].eq(1)
        one2zero=torch.sum(count)
        FCR=one2zero/int(0.1*percent*total)
        f = open('{}/FCR({}%).txt'.format(args.save,percent*10), 'a')
        f.write(str(epoch) + ' '+'one2zero='+str(one2zero) + ' '+'FCR='+str(FCR) + '\n')
        f.close()
    
def test(epoch):
    model.eval()
    ###################################### train_loader
    test_loss = 0
    correct = 0
    for data, target in train_loader:
        if args.cuda:
            data, target = data.to(device), target.to(device)
            #data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data.item()# sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(train_loader.dataset)
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    
    prec_train_loader=correct / float(len(train_loader.dataset))
    
    f = open('{}/{}-L1-trainset-accuracy.txt'.format(args.save,args.arch), 'a')
    f.write(str(epoch) + '  '+'train set Accuracy:'+str(prec_train_loader) + '  '+'train set Loss:'+str(test_loss) + '\n')
    f.close()
    #############################################   test_loader
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.to(device), target.to(device)
            #data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data.item()# sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    prec_test_loader=correct / float(len(test_loader.dataset))
    ###############################
    f = open('{}/{}-L1-testset-accuracy.txt'.format(args.save,args.arch), 'a')
    f.write(str(epoch) + '  '+'test set Accuracy:'+str(prec_test_loader) + ' '+'test set Loss:'+str(test_loss) + '\n')
    f.close()
    return prec_test_loader

def save_checkpoint(state, is_best, epoch, filepath):
    filename ='{}epoch-L1.pth.tar'.format(epoch)
    torch.save(state, os.path.join(filepath, filename))
    if is_best:
        shutil.copyfile(os.path.join(filepath, filename), os.path.join(filepath, 'model_best-L1.pth.tar'))

if __name__=="__main__":
# Training settings
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                        help='train with channel sparsity regularization')
    parser.add_argument('--s', type=float, default=0.0001,
                        help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--refine', default='', type=str, metavar='PATH',
                        help='path to the pruned model to be fine tuned')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for tr aining (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--epochs', type=int, default=160, metavar='N',
                        help='number of epochs to train (default: 160)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save', default='./', type=str, metavar='PATH',
                        help='path to save prune model (default: current directory)')
    parser.add_argument('--arch', default='vgg', type=str,
                        help='architecture to use')
    parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
    parser.add_argument('--depth', default=19, type=int,
                        help='depth of the neural network')
    parser.add_argument('--gpu', default=0, type=int)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.sr:
        args.save='./{}-{}/sparsity'.format(args.arch,args.dataset)
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    device = torch.device("cuda:{}".format(args.gpu))
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.Pad(4),
                               transforms.RandomCrop(32),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.Pad(4),
                               transforms.RandomCrop(32),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

    if args.refine:
        checkpoint = torch.load(args.refine)
        model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
    if args.cuda:       
        model=model.to(device)
        #model.cuda()
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                  .format(args.resume, checkpoint['epoch'], best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]
    mask1 = torch.zeros(9,total)
    mask2 = torch.ones(9,total)
    mask=torch.zeros(9,total)

    best_prec1 = 0.
    all_time=0
    for epoch in range(args.start_epoch, args.epochs):
        if epoch in [args.epochs*0.5, args.epochs*0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        start_time=time.time()
        train(epoch)
        end_time=time.time()
        epoch_time=end_time-start_time
        all_time+=epoch_time
        prec1 = test(epoch)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, epoch, filepath=args.save)

    f = open('{}/{}-L1-testset-accuracy.txt'.format(args.save, args.arch), 'a')
    f.write(str(epoch) + '  '+"Best accuracy:="+str(best_prec1) + '\n')
    f.close()