import torch
from torch import Tensor
from torch.nn import Module
import numpy as np


class MaskedBMLoss(Module):

    def __init__(self, loss_fn: Module):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, pred: Tensor, true: Tensor, n_frames: Tensor):
        loss = []
        for i, frame in enumerate(n_frames):
            loss.append(self.loss_fn(pred[i, :, :frame], true[i, :, :frame]))
        return torch.mean(torch.stack(loss))


class MaskedFrameLoss(Module):

    def __init__(self, loss_fn: Module):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, pred: Tensor, true: Tensor, n_frames: Tensor):
        # input: (B, T)
        loss = []
        for i, frame in enumerate(n_frames):
            #print(frame)
            loss.append(self.loss_fn(pred[i, :, :frame], true[i, :, :frame]))
            #print(pred[i, :, :frame].shape)
        return torch.mean(torch.stack(loss))

class MaskedFrameLoss2(Module):

    def __init__(self, loss_fn: Module):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, pred: Tensor, true: Tensor, n_frames: Tensor):
        # input: (B, T)
        loss = []
        for i, frame in enumerate(n_frames):
            #print(frame)
            #print(pred.shape)
            loss.append(self.loss_fn(pred[i, :frame], true[i, :frame]))
            #print(pred[i, :, :frame].shape)
        return torch.mean(torch.stack(loss))


class MaskedLoss(Module):

    def __init__(self, loss_fn: Module):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, pred: Tensor, true: Tensor, n_frames: Tensor):
        # input: (B, T)
        loss = []
        for i, frame in enumerate(n_frames):
            loss.append(self.loss_fn(pred[i, :frame].mean(-1), true[i]))
        return torch.mean(torch.stack(loss))


class MaskedContrastLoss(Module):

    def __init__(self, margin: float = 0.99):
        super().__init__()
        self.margin = margin

    def forward(self, pred1: Tensor, pred2: Tensor, labels: Tensor, n_frames: Tensor):
        # input: (B, C, T)
        loss = []
        for i, frame in enumerate(n_frames):
            # mean L2 distance squared
            d = torch.dist(pred1[i, :, :frame], pred2[i, :, :frame], 2)
            if labels[i] == 0:
                # if is positive pair, minimize distance
                loss.append(d ** 2 / (128 * frame))
            else:
                # if is negative pair, minimize (margin - distance) if distance < margin
                loss.append(torch.clip(self.margin - d, min=0.) ** 2 / (128 * frame))

        '''
        for i, frame in enumerate(n_frames):
            # mean L2 distance squared
            for n in frame:
                d = torch.dist(pred1[i, :, n], pred2[i, :, n], 2)
                if labels[i][n]:
                    # if is positive pair, minimize distance
                    loss.append(d ** 2)
                else:
                    # if is negative pair, minimize (margin - distance) if distance < margin
                    loss.append(torch.clip(self.margin - d, min=0.) ** 2)
        '''

        return torch.mean(torch.stack(loss))

class FrameMaskedContrastLoss2(Module):

    def __init__(self, margin: float = 0.99):
        super().__init__()
        self.margin = margin

    def forward(self, pred1: Tensor, pred2: Tensor, labels: Tensor, n_frames: Tensor):
        # input: (B, C, T)
        loss = []
        #print(labels.shape)
        
        batch, _, _= pred1.shape

        for i, frame in enumerate(n_frames):
            #print(frame)
            # mean L2 distance squared
            #loss = []
            for n in range(frame):
                d = torch.dist(pred1[i, :, n], pred2[i, :, n], 2) * frame
                #if i == 0 and n == 100:
                #    print('distance', d)
                if labels[i][n]:
                    # if is positive pair, minimize distance
                    loss.append(d ** 2)
                else:
                    # if is negative pair, minimize (margin - distance) if distance < margin
                    loss.append(torch.clip(self.margin - d, min=0.) ** 2)

        return torch.mean(torch.stack(loss))

class FrameMaskedContrastLoss(Module):

    def __init__(self, margin: float = 0.99):
        super().__init__()
        self.margin = margin

    def forward(self, pred1: Tensor, pred2: Tensor, labels: Tensor, n_frames: Tensor):
        # input: (B, C, T)
        loss = []
        #print(labels.shape)
        
        batch, _, _= pred1.shape

        for i, frame in enumerate(n_frames):
            #print(frame)
            # mean L2 distance squared
            #loss = []
            for n in range(frame):
                d = torch.dist(pred1[i, :, n], pred2[i, :, n], 2)
                #if i == 0 and n == 100:
                #    print('distance', d)
                if labels[i][n]:
                    # if is positive pair, minimize distance
                    loss.append(d ** 2)
                else:
                    # if is negative pair, minimize (margin - distance) if distance < margin
                    loss.append(torch.clip(self.margin - d, min=0.) ** 2)

        return torch.sum(torch.stack(loss)) / batch

class TemporalContrastiveLoss(Module):

    def __init__(self, margin: float = 0.99):
        super().__init__()
        self.margin = margin

    def forward(self, pred: Tensor, label: Tensor, n_frames: Tensor):
        loss = []
        
        a1 = 0.1
        a2 = 0.03
        for i, frame in enumerate(n_frames):
            z = pred[i, :frame]
            y = label[i, :frame]
            #z_norm = torch.sqrt(torch.sum(z*z,dim = -1,keepdim=True))
            #z = z/z_norm
            
            #print(y.shape)
            T = y.shape[0]
            y_diff = y[1:T]-y[0:T-1] #b*156
            y_l1 = torch.abs(y_diff) #b*155
            #zero = torch.zeros_like(z_l1)
            one = torch.ones_like(y_l1) 
            y_l1_a1 = torch.where(y_l1>0,one,y_l1) #b*155
            y_l1_a2 = torch.abs(y_l1_a1-1)

            z_diff = z[1:T]-z[0:T-1]
            #z_l2 = z[:T-1]*z[:T-1]
            z_l2 = z_diff ** 2
            z_l1 = torch.abs(z_diff)
            
            loss_i = a2 * torch.sum(z_l2*y_l1_a2) + a1 * torch.sum(torch.clip((self.margin - z_l1), min=0.) ** 2 * y_l1_a1)

            loss.append(loss_i)

        return torch.mean(torch.stack(loss))

class TemporalContrastiveLoss_sig(Module):

    def __init__(self):
        super().__init__()
        
        self.sigmoid = torch.nn.Sigmoid()        

    def forward(self, pred: Tensor, label: Tensor, n_frames: Tensor):
        loss = []
        
        a1 = 0.1
        a2 = 0.03
        
        pred = self.sigmoid(pred)

        for i, frame in enumerate(n_frames):
            z = pred[i, :frame]
            y = label[i, :frame]
            #z_norm = torch.sqrt(torch.sum(z*z,dim = -1,keepdim=True))
            #z = z/z_norm
            
            #print(y.shape)
            T = y.shape[0]
            y_diff = y[1:T]-y[0:T-1] #b*156
            y_l1 = torch.abs(y_diff) #b*155
            #zero = torch.zeros_like(z_l1)
            one = torch.ones_like(y_l1) 
            y_l1_a1 = torch.where(y_l1>0,one,y_l1) #b*155
            y_l1_a2 = torch.abs(y_l1_a1-1)

            z_diff = z[1:T]-z[0:T-1]
            #z_l2 = z[:T-1]*z[:T-1]
            z_l2 = z_diff ** 2
            
            loss_i = a2 * torch.sum(z_l2[:T-1]*y_l1_a2[:T-1]) - a1 * torch.sum(z_l2[:T-1]*y_l1_a1[:T-1])

            loss.append(loss_i)

        return torch.mean(torch.stack(loss))


def TEMPORALCONTRASTIVELOSS(z,y):
    y = y.permute(0,2,1)
    a1=0.1
    a2=0.03
    z_norm =torch.sqrt(torch.sum(z*z,dim = -1,keepdim=True))
    z = z/z_norm
    #aa = torch.sqrt(torch.sum(z*z,dim = -1,keepdim=True))
    #print(aa)
    
    #print(z.shape)
    #print(y.shape)
    B,T,_ = y.shape
    y[:,0:T-1,:] = y[:,1:T,:]-y[:,0:T-1,:] #b*156
    y_l1 =torch.sum(torch.abs(y[:,:T-1,:]),dim= 2,keepdim=True) #b*155
    #zero = torch.zeros_like(z_l1)
    one = torch.ones_like(y_l1) 
    y_l1_a1 =torch.where(y_l1>0,one,y_l1) #b*155
    y_l1_a2 = torch.abs(y_l1_a1-1)

    z[:,0:T-1,:] = z[:,1:T,:]-z[:,0:T-1,:]
    z_l2 = torch.sum(z[:,:T-1,:]*z[:,:T-1,:],dim= 2,keepdim=True)
    
    return ( a2*torch.sum(z_l2[:,:T-1,:]*y_l1_a2[:,:T-1,:],dim=(0,1,2))-a1*torch.sum(z_l2[:,:T-1,:]*y_l1_a1[:,:T-1,:],dim=(0,1,2)))/40


