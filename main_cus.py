#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import sys, os
import argparse
import numpy as np

from attacker.pgd import Linf_PGD

parser = argparse.ArgumentParser(description='PyTorch CustomSmoothingLoss Training')
parser.add_argument('--lr', required=True, type=float, help='learning rate')
parser.add_argument('--mixalpha', required=True, type=float, help='mixup alpha')
parser.add_argument('--augsigma', required=True, type=float, help='augmenation sigma')
parser.add_argument('--data', required=True, type=str, help='dataset name')
parser.add_argument('--model', required=True, type=str, help='model name')
parser.add_argument('--root', required=True, type=str, help='path to dataset')
parser.add_argument('--model_out', required=True, type=str, help='output path')
parser.add_argument('--resume', action='store', help='Resume training')
parser.add_argument('--resume_from', type=str,  help='Resume training')
opt = parser.parse_args()
# Data
print('==> Preparing data..')
if opt.data == 'cifar10':
    nclass = 10
    img_width = 32
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root=opt.root, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=opt.root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
elif opt.data == 'restricted_imagenet':
    data_path = os.path.expandvars(opt.data)
    dataset = DATASETS[opt.data](opt.root)

    train_loader, val_loader = dataset.make_loaders(2,128, data_aug= True)

    trainloader = helpers.DataPrefetcher(train_loader)
    testloader = helpers.DataPrefetcher(val_loader)
    nclass = 10
    img_width = 224
elif opt.data == 'mnist':
    trainloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor()),batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('../data', train=False,transform=transforms.ToTensor()),batch_size=100, shuffle=False, num_workers=2)
else:
    raise NotImplementedError('Invalid dataset')

# Model
if opt.model == 'vgg':
    from models.vgg import VGG
    #net = nn.DataParallel(VGG('VGG16', nclass, img_width=img_width).cuda())
    net = VGG('VGG16', nclass, img_width=img_width).cuda()
elif opt.model == 'aaron':
    from models.aaron import Aaron
    net = nn.DataParallel(Aaron(nclass).cuda())
elif opt.model == 'resnet':
    model_ft = resnet50(pretrained=False,num_classes=10)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 10)
    net = model_ft.cuda()
    #net = nn.DataParallel(model_ft.cuda())
elif opt.model == 'wide_resnet':
    from models.wideresnet import *
    net= nn.DataParallel(WideResNet().cuda())
    net = model_ft.cuda()
else:
    raise NotImplementedError('Invalid model')


if opt.resume and opt.resume_from:
    print(f'==> Resume from {opt.resume_from}')
    net.load_state_dict(torch.load(opt.resume_from))

net = nn.DataParallel(net)

#cudnn.benchmark = True

# Loss function
criterion = nn.CrossEntropyLoss()

# label smoothing
def LabelSmoothingLoss(outputs, targets):
    eps = 0.1
    batch_size, n_class = outputs.size()
    one_hot = torch.zeros_like(outputs).scatter(1,targets.view(-1,1),1)
    one_hot = one_hot*(1-eps) + (1-one_hot)*eps/ (n_class-1)
    log_prb = F.log_softmax(outputs,dim=1)
    loss = - (one_hot * log_prb).sum()/batch_size
    return loss

def CustomSmoothingLoss(outputs, targets, u):
    eps = 0.1
    batch_size, n_class = outputs.size()
    one_hot = torch.zeros_like(outputs).scatter(1,targets.view(-1,1),1)
    new_prob = (1-eps)*one_hot+eps*u
    new_prob /= new_prob.sum()
    log_prb = F.log_softmax(outputs,dim=1)
    loss = - (new_prob * log_prb).sum()/batch_size
    return loss

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

    #x = x + 0.1*torch.randn_like(x)
    x = x + opt.augsigma * torch.randn_like(x)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def custom_criterion(criterion, pred, y_a, y_b, lam, u):
    return lam * criterion(pred, y_a, u) + (1 - lam) * criterion(pred, y_b, u)


def dirismooth(outputs,targets,u):
    eps = 0.5
    one_hot = torch.zeros_like(outputs).scatter(1,targets.view(-1,1),1)
    alpha = one_hot * 10 + u
    alpha.clamp_(min=0)
    #print(alpha)
    distri = torch.distributions.Dirichlet(alpha) 
    batch_size, n_class = outputs.size()
    #one_hot = one_hot*(1-eps) + distri.rsample()*eps/n_class
    #one_hot = distri.sample()
    log_prb = F.log_softmax(outputs,dim=1)
    #loss = - (one_hot * log_prb).sum()/batch_size
    loss = - (distri.rsample() * log_prb).sum()/batch_size
    return loss 
        

def train(epoch, u):
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    aug = 3
    folds = 2
    alpha = opt.mixalpha *torch.ones(folds)
    #aug = 1 
    # label smoothing alpha
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        for k in range(aug):
            optimizer_net.zero_grad()
            inputs_mix, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                      opt.mixalpha)
            outputs = net(inputs_mix)
            loss = custom_criterion(dirismooth, outputs, targets_a, targets_b, lam, u )
            loss.backward()
            #print(u.grad)
            optimizer_net.step()
            pred = torch.max(outputs, dim=1)[1]
            correct += (lam * pred.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * pred.eq(targets_b.data).cpu().sum().float())

            total += targets.numel()
            train_loss += loss 
        if batch_idx%500==0:
            print(f'[TRAIN] {batch_idx} {loss:.3f} Acc {100.*correct/total:.3f}')
        optimizer_u.zero_grad()
        #inputs = inputs + opt.augsigma * torch.randn_like(inputs)
        outputs = net(inputs)
        loss = -dirismooth(outputs, targets, u)
        loss.backward()
        #print(u.grad)
        optimizer_u.step()
    print(f'[TRAIN] Loss: {train_loss:.3f}')
    print(f'[TRAIN] Acc: {100.*correct/total:.3f}')


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print(f'[TEST] Acc: {100.*correct/total:.3f}')
    
    robust_acc = test_attack()
    acc = 100.*correct /total
    if robust_acc > best_acc:
        best_acc = robust_acc
    # Save checkpointafter each epoch
        torch.save(net.module.state_dict(), opt.model_out)
# Go
def ensemble_inference(x_in):
    batch = x_in.size(0)
    prev = 0
    prob = torch.FloatTensor(batch, 10).zero_().cuda()
    answer = []
    with torch.no_grad():
        for n in range(1):
            for _ in range(n - prev):
                p = softmax(net(x_in))
                prob.add_(p)
            answer.append(prob.clone())
            prev = n
        for i, a in enumerate(answer):
            answer[i] = torch.max(a, dim=1)[1]
    return answer

    

def test_attack():
    correct = 0 
    total = 0
    #max_iter = 100
    distortion = 0
    batch = 0
    eps = 0.03
    for it, (x, y) in enumerate(testloader):
        x, y = x.cuda(), y.cuda()
        x_adv = Linf_PGD(x, y, net, 5, eps)
        pred = torch.max(net(x_adv),dim=1)[1]
        correct += torch.sum(pred.eq(y)).item()
        total += y.numel()
        batch += 1
    
    correct = str(correct / total)
    #print(f'{distortion/batch},' + ','.join(correct))
    print(f'{eps},' + correct)
    return float(correct)


if opt.data == 'cifar10':
    n_class = 10
    epochs = [80, 60, 40, 20]
elif opt.data == 'restricted_imagenet':
    epochs = [30, 20, 20, 10]
elif opt.data == 'mnist':
    epochs = [60, 40, 20]
    n_class = 10
count = 0


u = torch.ones(nclass).cuda()
u.requires_grad= True
best_acc = 0
for epoch in epochs:
    optimizer_net = SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5.0e-4)
    #optimizer_u = SGD([u], lr=opt.lr*0.2, momentum=0.9, weight_decay=5.0e-4)
    #optimizer_u = SGD([u], lr=opt.lr*5, momentum=0.9, weight_decay=5.0e-4)
    optimizer_u = SGD([u], lr=opt.lr, momentum=0.9, weight_decay=5.0e-4)
    for _ in range(epoch):
        train(count,u)
        test(count)
        #test_attack()
        count += 1
    opt.lr /= 10
