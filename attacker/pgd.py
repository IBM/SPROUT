import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from .linf_sgd import Linf_SGD
from torch.optim import SGD, Adam

# performs Linf-constraint PGD attack w/o noise
# @epsilon: radius of Linf-norm ball

def random_start(x,eps):
    x+=torch.FloatTensor(x.size()).uniform_(-eps,eps).cuda()
    x.clamp_(0,1)
    return x

def Linf_PGD(x_nat, y_true, net, steps, eps, imagenet=False, random=False, cw=False):
    if eps == 0:
        return x_nat
    training = net.training
    if training:
        net.eval()
    x_in = x_nat.clone()
    x_adv = random_start(x_in,eps)
    x_adv = x_in
    x_adv.requires_grad=True
    optimizer = Linf_SGD([x_adv], lr=0.007)
    zero = torch.FloatTensor([0]).cuda()
    #optimizer = SGD([x_adv], lr=0.01)
    for _ in range(steps):
        optimizer.zero_grad()
        net.zero_grad()
        if imagenet:
            out = net(x_adv)
        else:
            out = net(x_adv)
        if cw:
            index = y_true.cpu().view(-1, 1)
            label_onehot = torch.FloatTensor(x_in.size(0), 200).zero_().scatter_(1, index, 1).cuda()
            real = torch.max(out*label_onehot -(1-label_onehot)*100000, dim=1)[0]
            #real = torch.sum(out*label_onehot, dim=1)
            other = torch.max(torch.mul(out, (1-label_onehot))-label_onehot*100000, 1)[0]
            loss = torch.sum(torch.max(real - other, zero))
        else:
            if random:
                loss = F.cross_entropy(out, y_true)
            else:
                loss = -F.cross_entropy(out, y_true)
        loss.backward()

        optimizer.step()
        diff = x_adv - x_nat
        diff.clamp_(-eps, eps)
        if imagenet:
            x_adv.detach().copy_(diff + x_nat)
            x_adv[:,0,:,:].data.clamp_(-0.485/0.229,(1-0.485)/0.229) 
            x_adv[:,1,:,:].data.clamp_(-0.456/0.224,(1-0.456)/0.224) 
            x_adv[:,2,:,:].data.clamp_(-0.406/0.225,(1-0.406)/0.225) 
            #x_adv.data[:,0,:,:]=(diff + x_in).data[:,0,:,:].clamp_(-0.485/0.229,(1-0.485)/0.229) 
            #x_adv.data[:,1,:,:]=(diff + x_in).data[:,1,:,:].clamp_(-0.456/0.224,(1-0.456)/0.224) 
            #x_adv.data[:,2,:,:]=(diff + x_in).data[:,2,:,:].clamp_(-0.406/0.225,(1-0.406)/0.225) 
        else:
            x_adv.detach().copy_((diff + x_nat).clamp_(0, 1))
    net.zero_grad()
    # reset to the original state
    if training:
        net.train()
    return x_adv

def L2_PGD(x_in, y_true, net, steps, eps):
    if eps == 0:
        return x_in
    training = net.training
    if training:
        net.eval()
    x_in = random_start(x_in,eps)
    x_adv = x_in.clone().requires_grad_(True)
    #optimizer = Adam([x_adv], lr=0.01)
    lr = 0.1
    for _ in range(steps):
        #optimizer.zero_grad()
        net.zero_grad()
        out, _ = net(x_adv)
        loss = -F.cross_entropy(out, y_true)
        loss.backward(retain_graph=True)
        #optimizer.step()
        grad = x_adv.grad
        #print(grad.size(),loss)
        g_norm = torch.norm(grad.view(grad.shape[0],-1),dim=1).view(-1,1,1,1)
        scaled_grad = grad/ (g_norm+ 1e-10)
        x_adv.data = x_adv.data - lr*scaled_grad
        diff = x_adv.data - x_in
        diff = diff.renorm(p=2, dim=0, maxnorm=eps)
        x_adv.data = (diff + x_in).clamp_(0,1)
        x_adv.grad.data.zero_()
    net.zero_grad()
    if training:
        net.train()
    return x_adv
# performs L2-constraint PGD attack w/o noise
# @epsilon: radius of L2-norm ball
#def L2_PGD(x_in, y_true, net, steps, eps):
#    if eps == 0:
#        return x_in
#    training = net.training
#    if training:
#        net.eval()
#    x_in = random_start(x_in,eps)
#    x_adv = x_in.clone().requires_grad_()
#    optimizer = Adam([x_adv], lr=0.01)
#    eps = torch.tensor(eps).view(1,1,1,1).cuda()
#    #print('====================')
#    for _ in range(steps):
#        optimizer.zero_grad()
#        net.zero_grad()
#        out, _ = net(x_adv)
#        loss = -F.cross_entropy(out, y_true)
#        loss.backward()
#        #print(loss.item())
#        optimizer.step()
#        diff = x_adv - x_in
#        norm = torch.sqrt(torch.sum(diff * diff, (1, 2, 3)))
#        norm = norm.view(norm.size(0), 1, 1, 1)
#        norm_out = torch.min(norm, eps)
#        diff = diff / norm * norm_out
#        x_adv.detach().copy_((diff + x_in).clamp_(0, 1))
#    net.zero_grad()
#    # reset to the original state
#    if training :
#        net.train()
#    return x_adv
def distance(x_adv, x):
    diff = (x_adv - x).view(x.size(0), -1)
    out = torch.max(torch.abs(diff), 1)[0]
    return out


def Linf_PGD_new(x_nat, y_true, net, steps, eps, imagenet=False, random=False):
    #mask = torch.LongTensor(x_nat.size(0)).cuda().zero_() 
    eps_l = np.arange(0.005,0.03,0.005)  
    training = net.training
    if training:
        net.eval()
    x_in = x_nat.clone()
    x_adv = x_in.clone()
    x_adv.requires_grad=True
    optimizer = Linf_SGD([x_adv], lr=0.007)
    for eps in eps_l:
        x_adv.data = x_nat.data.clone()
        for _ in range(steps):
            optimizer.zero_grad()
            net.zero_grad()
            if imagenet:
                out = net(x_adv)
            else:
                out, _ = net(x_adv)
            loss = -F.cross_entropy(out, y_true)
            loss.backward()

            optimizer.step()
            diff = x_adv - x_nat
            diff.clamp_(-eps, eps)
            if imagenet:
                x_adv.detach().copy_(diff + x_nat)
                x_adv[:,0,:,:].data.clamp_(-0.485/0.229,(1-0.485)/0.229) 
                x_adv[:,1,:,:].data.clamp_(-0.456/0.224,(1-0.456)/0.224) 
                x_adv[:,2,:,:].data.clamp_(-0.406/0.225,(1-0.406)/0.225) 
                #x_adv.data[:,0,:,:]=(diff + x_in).data[:,0,:,:].clamp_(-0.485/0.229,(1-0.485)/0.229) 
                #x_adv.data[:,1,:,:]=(diff + x_in).data[:,1,:,:].clamp_(-0.456/0.224,(1-0.456)/0.224) 
                #x_adv.data[:,2,:,:]=(diff + x_in).data[:,2,:,:].clamp_(-0.406/0.225,(1-0.406)/0.225) 
            else:
                x_adv.detach().copy_((diff + x_nat).clamp_(0, 1))
        net.zero_grad()
        mask = (torch.max(net(x_adv)[0],dim=1)[1] == y_true)
        if mask is None:
            break
        x_in.data[mask] = x_adv.data[mask]
        #print(distance(x_in,x_nat))
    # reset to the original state
    if training:
        net.train()
    return x_in

criterion = nn.CrossEntropyLoss(reduction='none')
def AdaptiveLoss(outputs, targets, dis):
    loss = criterion(outputs, targets)
    factor = torch.log2(2-20*dis)
    #print(loss,factor,dis)
    return torch.mean(factor*loss)



def Linf_PGD_weight(x_nat, y_true, net, steps, eps, imagenet=False, random=False):
    #mask = torch.LongTensor(x_nat.size(0)).cuda().zero_() 
    eps_l = np.arange(0.005,0.03,0.005)  
    training = net.training
    if training:
        net.eval()
    x_in = x_nat.clone()
    x_adv = x_in.clone()
    dis = torch.zeros(y_true.size()).cuda()
    x_adv.requires_grad=True
    optimizer = Linf_SGD([x_adv], lr=0.007)
    for eps in eps_l:
        x_adv.data = x_nat.data.clone()
        for _ in range(steps):
            optimizer.zero_grad()
            net.zero_grad()
            if imagenet:
                out = net(x_adv)
            else:
                out, _ = net(x_adv)
            if random:
                loss = -F.cross_entropy(out, y_true)
            else:
                loss = -AdaptiveLoss(out, y_true, dis)
            loss.backward()

            optimizer.step()
            diff = x_adv - x_nat
            diff.clamp_(-eps, eps)
            if imagenet:
                x_adv.detach().copy_(diff + x_nat)
                x_adv[:,0,:,:].data.clamp_(-0.485/0.229,(1-0.485)/0.229) 
                x_adv[:,1,:,:].data.clamp_(-0.456/0.224,(1-0.456)/0.224) 
                x_adv[:,2,:,:].data.clamp_(-0.406/0.225,(1-0.406)/0.225) 
                #x_adv.data[:,0,:,:]=(diff + x_in).data[:,0,:,:].clamp_(-0.485/0.229,(1-0.485)/0.229) 
                #x_adv.data[:,1,:,:]=(diff + x_in).data[:,1,:,:].clamp_(-0.456/0.224,(1-0.456)/0.224) 
                #x_adv.data[:,2,:,:]=(diff + x_in).data[:,2,:,:].clamp_(-0.406/0.225,(1-0.406)/0.225) 
            else:
                x_adv.detach().copy_((diff + x_nat).clamp_(0, 1))
        net.zero_grad()
        mask = (torch.max(net(x_adv)[0],dim=1)[1] == y_true)
        if mask is None:
            break
        x_in.data[mask] = x_adv.data[mask]
        dis[mask] = eps
        #print(distance(x_in,x_nat))
        # reset to the original state
    if training:
        net.train()
    return x_in



def Linf_PGD_so(x_nat, y_true, net, steps, eps, imagenet=False, random=False, cw=False):
    #mask = torch.LongTensor(x_nat.size(0)).cuda().zero_() 
    #eps_l = np.arange(0.005,0.03,0.005)  
    training = net.training
    if training:
        net.eval()
    x_in = x_nat.clone()
    x_adv = x_in.clone()
    x_adv.requires_grad=True
    optimizer = Linf_SGD([x_adv], lr=0.007)
    eps += 0.005
    eps.clamp_(max=0.03)
    #sorted_eps, perm = torch.sort(eps)
    #for eps in eps_l:
    x_adv.data = x_nat.data.clone()
    zero = torch.tensor([0.0]).cuda()
    for _ in range(steps):
        optimizer.zero_grad()
        net.zero_grad()
        if imagenet:
            out = net(x_adv)
        else:
            out = net(x_adv)
        if cw:
            real = torch.max(torch.mul(out, y_true), 1)[0]
            other = torch.max(torch.mul(out, (1-y_true))-y_true*10000, 1)[0]
            loss = torch.sum(torch.max(real - other, zero))
        else:
            log_prb = F.log_softmax(out,dim=1)
        #print(log_prb.shape, y_true.shape)
            loss = (y_true * log_prb).sum()/x_adv.size(0)
        loss.backward()

        optimizer.step()
        diff = x_adv - x_nat
        #diff.clamp_(-eps, eps)
        diff = torch.max(diff,-torch.ones_like(diff)*eps.view(-1,1,1,1).expand_as(diff))
        diff = torch.min(diff,torch.ones_like(diff)*eps.view(-1,1,1,1).expand_as(diff))
        if imagenet:
            x_adv.detach().copy_(diff + x_nat)
            x_adv[:,0,:,:].data.clamp_(-0.485/0.229,(1-0.485)/0.229) 
            x_adv[:,1,:,:].data.clamp_(-0.456/0.224,(1-0.456)/0.224) 
            x_adv[:,2,:,:].data.clamp_(-0.406/0.225,(1-0.406)/0.225) 
            #x_adv.data[:,0,:,:]=(diff + x_in).data[:,0,:,:].clamp_(-0.485/0.229,(1-0.485)/0.229) 
            #x_adv.data[:,1,:,:]=(diff + x_in).data[:,1,:,:].clamp_(-0.456/0.224,(1-0.456)/0.224) 
            #x_adv.data[:,2,:,:]=(diff + x_in).data[:,2,:,:].clamp_(-0.406/0.225,(1-0.406)/0.225) 
        else:
            x_adv.detach().copy_((diff + x_nat).clamp_(0, 1))
    net.zero_grad()
    mask = (torch.max(net(x_adv),dim=1)[1] == torch.max(y_true,dim=1)[1])
    if mask is None:
        return x_in
    x_in.data[mask] = x_adv.data[mask]
        #print(distance(x_in,x_nat))
    # reset to the original state
    if training:
        net.train()
    return x_in



def Linf_PGD_so_cw(x_nat, y_true, net, steps, eps, one_hot, imagenet=False, random=False, cw=False):
    #mask = torch.LongTensor(x_nat.size(0)).cuda().zero_() 
    #eps_l = np.arange(0.005,0.03,0.005)  
    training = net.training
    if training:
        net.eval()
    x_in = x_nat.clone()
    x_adv = x_in.clone()
    x_adv.requires_grad=True
    optimizer = Linf_SGD([x_adv], lr=0.002)
    eps += 0.0025
    eps.clamp_(max=0.03)
    #sorted_eps, perm = torch.sort(eps)
    #for eps in eps_l:
    x_adv.data = x_nat.data.clone()
    zero = torch.tensor([0.0]).cuda()
    for _ in range(steps):
        optimizer.zero_grad()
        net.zero_grad()
        if imagenet:
            out = net(x_adv)
        else:
            out = net(x_adv)
        if cw:
            real = torch.max(out*one_hot -(1-one_hot)*100000, dim=1)[0]
            other = torch.max(torch.mul(out, (1-one_hot))-one_hot*10000, 1)[0]
            loss1 = torch.max(real - other+50, zero)
            loss1 = torch.sum(loss1 * eps)
            log_prb = F.log_softmax(out,dim=1)
            #print(log_prb.shape, y_true.shape)
            loss2 = (y_true * log_prb).sum()/x_adv.size(0)
            loss = loss1 + loss2
        else:
            log_prb = F.log_softmax(out,dim=1)
        #print(log_prb.shape, y_true.shape)
            loss = (y_true * log_prb).sum()/x_adv.size(0)
        loss.backward()

        optimizer.step()
        diff = x_adv - x_nat
        #diff.clamp_(-eps, eps)
        diff = torch.max(diff,-torch.ones_like(diff)*eps.view(-1,1,1,1).expand_as(diff))
        diff = torch.min(diff,torch.ones_like(diff)*eps.view(-1,1,1,1).expand_as(diff))
        if imagenet:
            x_adv.detach().copy_(diff + x_nat)
            x_adv[:,0,:,:].data.clamp_(-0.485/0.229,(1-0.485)/0.229) 
            x_adv[:,1,:,:].data.clamp_(-0.456/0.224,(1-0.456)/0.224) 
            x_adv[:,2,:,:].data.clamp_(-0.406/0.225,(1-0.406)/0.225) 
            #x_adv.data[:,0,:,:]=(diff + x_in).data[:,0,:,:].clamp_(-0.485/0.229,(1-0.485)/0.229) 
            #x_adv.data[:,1,:,:]=(diff + x_in).data[:,1,:,:].clamp_(-0.456/0.224,(1-0.456)/0.224) 
            #x_adv.data[:,2,:,:]=(diff + x_in).data[:,2,:,:].clamp_(-0.406/0.225,(1-0.406)/0.225) 
        else:
            x_adv.detach().copy_((diff + x_nat).clamp_(0, 1))
    net.zero_grad()
    mask = (torch.max(net(x_adv),dim=1)[1] == torch.max(y_true,dim=1)[1])
    if mask is None:
        return x_in
    x_in.data[mask] = x_adv.data[mask]
        #print(distance(x_in,x_nat))
    # reset to the original state
    if training:
        net.train()
    return x_in
