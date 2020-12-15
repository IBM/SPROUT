#!/usr/bin/env python
import argparse
import torch, os
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
from torchvision.models import resnet50, resnet18
#from models.resnet import ResNet18
from models.vgg import VGG
from attacker.pgd import Linf_PGD, L2_PGD
from attacker.cw import cw
# arguments
parser = argparse.ArgumentParser(description='Bayesian Inference')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--defense', type=str, required=True)
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--root', type=str, required=True)
parser.add_argument('--n_ensemble', type=str, required=True)
parser.add_argument('--steps', type=int, required=True)
parser.add_argument('--max_norm', type=str, required=True)
parser.add_argument('--attack', type=str, default='Linf')
parser.add_argument('--alpha', type=float)
parser.add_argument('--model_dir', type=str)

opt = parser.parse_args()

opt.max_norm = [float(s) for s in opt.max_norm.split(',')]
opt.n_ensemble = [int(n) for n in opt.n_ensemble.split(',')]

# attack
if opt.attack == 'Linf':
    attack_f = Linf_PGD
elif opt.attack == 'L2':
    attack_f = L2_PGD
elif opt.attack == 'CW':
    attack_f = cw
else:
    raise ValueError(f'invalid attach function: {opt.attack}')


# dataset
#print('==> Preparing data..')
if opt.data == 'cifar10':
    nclass = 10
    img_width = 32
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root='../data/cifar10-batches-py/', train=True, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='../data/cifar10-batches-py/', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)
elif opt.data == 'stl10':
    nclass = 10
    img_width = 96
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        ])
    testset = torchvision.datasets.STL10(root=opt.root, split='test', transform=transform_test, download=True)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=100, shuffle=False)
elif opt.data == 'fashion':
    nclass = 10
    img_width = 28
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.FashionMNIST(root=opt.root, train=False, transform=transform_test, download=True)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=opt.batch_size, shuffle=False)
elif opt.data == 'tiny_imagenet':
    nclass = 200
    img_width = 64
    transform_train = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.ImageFolder(os.path.join(opt.root, 'train'), transform=transform_train)  
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.ImageFolder(os.path.join(opt.root, 'val'), transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
elif opt.data == 'restricted_imagenet':
    data_path = os.path.expandvars(opt.data)
    dataset = DATASETS[opt.data](opt.root)
    train_loader, val_loader = dataset.make_loaders(2,128, data_aug= True)
    train_sampler = train_loader.sampler
    trainloader = helpers.DataPrefetcher(train_loader)
    testloader = helpers.DataPrefetcher(val_loader)
    nclass = 10
    img_width = 224
else:
    raise ValueError(f'invlid dataset: {opt.data}')


# load model
if opt.model == 'vgg':
    if opt.defense in ('plain', 'adv','lb','mixup','lb_aug','aug','combine','cus','resume','trades'):
        from models.vgg import VGG
        net = nn.DataParallel(VGG('VGG16', nclass, img_width=img_width), device_ids=range(1))
        #net = VGG('VGG16', nclass, img_width=img_width)
        if opt.alpha and opt.defense=="mixup":
            #print('Attacking '+f'./checkpoint/{opt.data}_{opt.model}_{opt.defense}_{opt.alpha}.pth')
            net.load_state_dict(torch.load(f'./checkpoint/{opt.data}_{opt.model}_{opt.defense}_{opt.alpha}.pth'))
        elif opt.defense == 'cus':
            #print('Attacking '+f'./checkpoint/{opt.data}_{opt.model}_{opt.defense}_{opt.alpha}.pth')
            net.load_state_dict(torch.load(f'./checkpoint/{opt.data}_{opt.model}_{opt.defense}_{opt.alpha}.pth'))
        #else:
            #print('Attacking '+f'./checkpoint/{opt.data}_{opt.model}_{opt.defense}.pth')
            #net.load_state_dict(torch.load(f'./checkpoint/{opt.data}_{opt.model}_{opt.defense}.pth'))
    elif opt.defense in ('vi', 'adv_vi'):
        from models.vgg_vi import VGG
        net = nn.DataParallel(VGG(1.0, 1.0, 1.0, 'VGG16', nclass, img_width=img_width), device_ids=range(1))
    elif opt.defense in ('rse'):
        from models.vgg_rse import VGG
        net = nn.DataParallel(VGG('VGG16', nclass, 0.2, 0.1, img_width=img_width), device_ids=range(1))
elif opt.model == 'aaron':
    if opt.defense in ('plain', 'adv','resume'):
        from models.aaron import Aaron
        net = nn.DataParallel(Aaron(nclass), device_ids=range(1))
    elif opt.defense in ('vi', 'adv_vi'):
        from models.aaron_vi import Aaron
        net = nn.DataParallel(Aaron(1.0, 1.0, 1.0, nclass), device_ids=range(1))
    elif opt.defense in ('rse'):
        from models.aaron_rse import Aaron
        net = nn.DataParallel(Aaron(nclass, 0.2, 0.1), device_ids=range(1))
elif opt.model == 'resnet':
    model_ft = resnet50(pretrained=False,num_classes=10)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 10)
    net = model_ft.cuda()
    net= nn.DataParallel(net)
elif opt.model == 'resnet18':
    model = resnet18()
    model.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    model.maxpool = nn.Sequential()
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = torch.nn.Linear(in_features=512,out_features=200,bias=True)
    net = model.cuda()
    net= nn.DataParallel(net)
elif opt.model == 'wide_resnet':
    from models.wideresnet import *
    device = torch.device("cuda")
    net = WideResNet().to(device)
    net = nn.DataParallel(net, device_ids=range(1))
else:
    raise ValueError('invalid opt.model')
#net.load_state_dict(torch.load(f'./checkpoint/{opt.data}_{opt.model}_{opt.defense}.pth'))
if opt.model_dir is not None:
    print('Attacking '+f'{opt.model_dir}')
    net.load_state_dict(torch.load(opt.model_dir))
#net = nn.DataParallel(net, device_ids=range(1))
#nclass = 10
net.cuda()
net.eval() # must set to evaluation mode
loss_f = nn.CrossEntropyLoss()
softmax = nn.Softmax(dim=1)
cudnn.benchmark = True


def ensemble_inference(x_in):
    batch = x_in.size(0)
    prev = 0
    prob = torch.FloatTensor(batch, nclass).zero_().cuda()
    answer = []
    with torch.no_grad():
        for n in opt.n_ensemble:
            for _ in range(n - prev):
                p = softmax(net(x_in))
                prob.add_(p)
            answer.append(prob.clone())
            prev = n
        for i, a in enumerate(answer):
            answer[i] = torch.max(a, dim=1)[1]
    return answer
    

def distance(x_adv, x):
    diff = (x_adv - x).view(x.size(0), -1)
    if opt.attack in ('CW', 'L2'):
        out = torch.sqrt(torch.sum(diff * diff) / x.size(0)).item()
        return out
    elif opt.attack in ('Linf'):
        out = torch.mean(torch.max(torch.abs(diff), 1)[0]).item()
        return out

# Iterate over test set
print('#norm, test accuracy')
for eps in opt.max_norm:

    correct = [0] * len(opt.n_ensemble)
    total = 0
    max_iter = 100
    distortion = 0
    batch = 0
    for it, (x, y) in enumerate(testloader):
        x, y = x.cuda(), y.cuda()
        #print(x.max(),x.min())
        x_adv = attack_f(x, y, net, opt.steps, eps, cw=False)
        pred = ensemble_inference(x_adv)
        for i, p in enumerate(pred):
            correct[i] += torch.sum(p.eq(y)).item()
        total += y.numel()
        distortion += distance(x_adv, x)
        batch += 1
        if it >= max_iter:
            break
    for i, c in enumerate(correct):
        correct[i] = c / total*100
    #print(f'testing {distortion/batch},' + ','.join(correct))
    print(correct[0])
