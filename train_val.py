#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function
import argparse
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models
import random
# import models
from utils import progress_bar
from torch.optim.lr_scheduler import StepLR
from choose_network import choose_network
from data_loader import get_train_valid_loader, get_test_loader
from torchsummary import summary
import pdb
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'
      

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default="resnet50", type=str,
                    help='model type (default: resnet50 |resnet101 | resnet121 )')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run')

parser.add_argument('--decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')

parser.add_argument('--dataset', default="CIFAR10", type=str,
                    help='CIFAR10|STL10')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--num_labels', type=int, default=4000)

parser.add_argument('--pretrained_from', type=str, default='scratch',
                    help='scratch | ImageNet_supervised | ImageNet_SimCLR | CIFAR10_SimCLR | MLCC_SimCLR | Simclrv2')

parser.add_argument('--net_from_name', type=bool, default=True)

parser.add_argument('--mixup', default=False, type=boolean_string,
                    help='')

parser.add_argument('--limit_data', default=False, type=boolean_string,
                    help='')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dataset_resolution', type= int , default= -1)
parser.add_argument('--save_root_dir', type= str , default= '/root/dataset2/finetuning_simclrv2')

args = parser.parse_args()
print(args.dataset_resolution)

use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
0
if args.seed != 0:
    torch.manual_seed(args.seed)


# Data
print('==> Preparing data..')


def get_SGD(net, name='SGD', lr=0.1, momentum=0.9, \
            weight_decay=5e-4, nesterov=True, bn_wd_skip=True):
    '''
    return optimizer (name) in torch.optim.
    If bn_wd_skip, the optimizer does not apply
    weight decay regularization on parameters in batch normalization.
    '''
    optim = getattr(torch.optim, name)
    decay = []
    no_decay = []
    for name, param in net.named_parameters():
        if ('bn' in name) and bn_wd_skip:
            no_decay.append(param)
        else:
            decay.append(param)

    per_param_args = [{'params': decay},
                      {'params': no_decay, 'weight_decay': 0.0}]

    optimizer = optim(per_param_args, lr=lr,
                      momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    return optimizer


# def get_transform_cifar(train=True):
#     if train:
#         return transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean['cifar10'], std['cifar10'])
#         ])
#     else:
#         return transforms.Compose([
#             transforms.ToTensor(), transforms.Normalize(mean['cifar10'], std['cifar10'])])


# def get_transform_MLCC(train=True):
#     if train:
#         return transforms.Compose([
#             transforms.RandomVerticalFlip(p=0.5),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.ToTensor(),
#             transforms.Normalize(mean['MLCC'], std['MLCC'])])
#     else:
#         return transforms.Compose([transforms.ToTensor(),
#                                    transforms.Normalize(mean['MLCC'], std['MLCC'])])


# transform_test = transforms.Compose([
#     transforms.Resize(size=(224, 224)),
#     transforms.ToTensor(),
#     # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# if args.dataset == "CIFAR10" or args.dataset == "STL10" or args.dataset == "SVHN":
#     print(args.dataset + "dataset\n")
#     trainset = datasets.CIFAR10(root='~/data', train=True, download=True, transform=get_transform_cifar(train=True))
#     testset = datasets.CIFAR10(root='~/data', train=False, download=True, transform=get_transform_cifar(train=False))
# elif args.dataset == "STL10":
#     print(args.dataset + "dataset\n")
#     trainset = datasets.STL10(root='~/data', train=True, download=True, transform=transform_train)
#     testset = datasets.STL10(root='~/data', train=False, download=True, transform=transform_test)
# elif args.dataset == "SVHN":
#     print(args.dataset + "dataset\n")
#     trainset = datasets.SVHN(root='~/data', train=True, download=True, transform=transform_train)
#     testset = datasets.SVHN(root='~/data', train=False, download=True, transform=transform_test)
# elif args.dataset == "MLCC":
#     print("MLCC Dataset\n")
#     trainset = torchvision.datasets.ImageFolder(root="/data/samsung/labeled/Train",
#                                                 transform=get_transform_MLCC(train=True))
#     testset = torchvision.datasets.ImageFolder(root="/data/samsung/labeled/Test",
#                                                transform=get_transform_MLCC(train=False))
    # trainset = torchvision.datasets.ImageFolder(root="/root/dataset2/Samsung_labeled_only/Train", transform=get_transform_MLCC(train=True))
    # testset = torchvision.datasets.ImageFolder(root="/root/dataset2/Samsung_labeled_only/Test", transform=get_transform_MLCC(train=False))

##################데이터 4000개만 쓰기########################--lr 0.002 --model resnet50 --net_from_name True --pretrained_from Simclrv2 --batch_size 64 --epoch 200 --decay 5e-4 --dataset CIFAR10 --num_classes 10 --limit_data True --num_labels 4000 --gpu 0 ###
# if args.limit_data == True:
#     trainset.data = trainset.data[:4000]
#     trainset.targets = trainset.targets[:4000]
#     print("only 4000 data")

#############################################################
trainloader, validateloder = get_train_valid_loader(data_dir= './data',
                                                    dataset = args.dataset,
                                                    batch_size = args.batch_size,
                                                    limit_data = args.limit_data,
                                                    num_labels = args.num_labels,
                                                    num_classes = args.num_classes,
                                                    augment = True,
                                                    random_seed = args.seed,
                                                    dataset_resolution= args.dataset_resolution,
                                                    valid_size = 0.1,
                                                    shuffle = True,
                                                    show_sample = False,
                                                    num_workers = 1,
                                                    pin_memory = True)

testloader = get_test_loader(data_dir = './data',
                             dataset = args.dataset,
                             batch_size = 1,
                             dataset_resolution = args.dataset_resolution,
                             shuffle = True,
                             num_workers = 1,
                             pin_memory= True)



# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7' + args.name + '_'
                            + str(args.seed))
    checkpoint = torch.load('./checkpoint/' + args.name + 'pt')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
else:
    print('==> Building model..')
    net = choose_network(args, net=args.model, pretrained_from=args.pretrained_from,
                         pretrained_model_dir='./pretrained_model')

    # if args.model == "ResNet50" :
    #     net = choose_network(args)
    # if args.pretrain == "supervised-imagenet":
    #     print("Supervised Imagenet Pretrained Resenet50")
    #     net = torchvision.models.resnet50(pretrained=True)
    #     tmp = net.fc.in_features
    #     net.fc = nn.Linear(tmp,10)
    # elif args.pretrain == "simCLR-imagenet" :
    #     # load resnet50 pretrained using SimCLR on imagenet
    #     print("SimCLR Imagenet Pretrained Resenet50")
    #     weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
    #     simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
    #     net = simclr.encoder
    #     net = torchvision.models.resnet50(pretrained=True)
    #     tmp = net.fc.in_features
    #     net.fc = nn.Linear(tmp,10)
    # elif args.pretrain == "scratch" :
    #     print("scratch Resenet50")
    #     net = torchvision.models.resnet50(pretrained=False)
    #     tmp = net.fc.in_features
    #     net.fc = nn.Linear(tmp,10)
    #
    # else :
    #     net = models.__dict__[args.model]()
if args.dataset_resolution == -1:
    dataset_resolution = size_same
else:
    dataset_resolution = f'size_{args.dataset_resolution}'
if args.limit_data == True:
    name = f'supervised_split_{args.pretrained_from}_{args.dataset}_{dataset_resolution}_{args.model}'
else:
    name = f'supervised_{args.pretrained_from}_{args.dataset}_{dataset_resolution}_{args.model}'
print("-------------------------")
print(name)
print("-------------------------")
model_dir = name + '_results'
save_dir =  os.path.join(args.save_root_dir, model_dir)
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
logname = (os.path.join(save_dir, 'log_' + name + '_'
                        + str(args.seed) + '.csv'))
if args.gpu is not None:
    torch.cuda.set_device(args.gpu)
if use_cuda:
    net.cuda()
    # net = torch.nn.DataParallel(net)
    print(torch.cuda.device_count())
    cudnn.benchmark = True
    print('Using CUDA..')

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
#                       weight_decay=args.decay)
# samsung setting below
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
optimizer = get_SGD(net, 'SGD', args.lr, args.momentum, args.weight_decay)
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay, nesterov=True)


# scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        if args.mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                           args.alpha, use_cuda)
            inputs, targets_a, targets_b = map(Variable, (inputs,
                                                          targets_a, targets_b))

        optimizer.zero_grad()
        outputs = net(inputs)  # kr
        if args.mixup:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)  # kr
        else:
            loss = criterion(outputs, targets)  # kr

        train_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        # correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
        #            + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
        correct += predicted.eq(targets.data).cpu().sum()

        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), reg_loss / (batch_idx + 1),
                        100. * correct / total, correct, total))
    return (train_loss / batch_idx, reg_loss / batch_idx, 100. * correct / total)


def val(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(validateloder):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        with torch.no_grad():
            inputs = Variable(inputs)
        target = Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(validateloder),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), 100. * correct / total,
                        correct, total))
    acc = 100. * correct / total
    
    if epoch == start_epoch + args.epoch - 1:
        PATH = f'{name}_last.pth'
        torch.save(net.state_dict(), os.path.join(save_dir, PATH))
    if acc > best_acc:
        PATH = f'{name}_epoch_{epoch}.pth'
        torch.save(net.state_dict(), os.path.join(save_dir, PATH))
        best_acc = acc
    return (test_loss / batch_idx, 100. * correct / total)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7' + args.name + '_'
               + str(args.seed))


def adjust_learning_rate(optimizer, epoch):
    ########################################################################
    # """decrease the learning rate at 100 and 150 epoch"""
    # lr = args.lr
    # if epoch >= 100:
    #     lr /= 10
    # if epoch >= 150:
    #     lr /= 10
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    ########################################################################
    # samsung setting below
    lr = args.lr
    if epoch % 50 == 0:
        lr /= 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                            'val loss', 'val acc'])

for epoch in range(start_epoch, args.epoch):
    # scheduler.step()
    # print('Epoch:', epoch, 'LR:', scheduler.get_lr())
    train_loss, reg_loss, train_acc = train(epoch)
    val_loss, val_acc = val(epoch)
    # adjust_learning_rate(optimizer, epoch)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, reg_loss, train_acc, val_loss,
                            val_acc])
