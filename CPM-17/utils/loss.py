
# -*- coding=utf-8 -*-

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LossFunction(nn.Module):

    def __init__(self):
        super(LossFunction, self).__init__()
        self.seg_ce_loss = Seg_CrossEntropyLoss()#CE Loss
        self.seg_dice_loss = Seg_DiceLoss() # Dice Loss 
        self.e_ce_loss = E_CrossEntropyLoss('gradient') # E Loss
        self.sub_loss = Sub_Neighbor_Loss(3) # T Loss
    def forward(self,inputs,target, ellipse, ellipse_gradient): 
        assert ellipse.size() == target.size()
        ellipse /= 255 # Normalization
        ellipse_gradient /= 255

        ce_loss = self.seg_ce_loss(inputs,target)

        input_soft = torch.softmax(inputs, dim=1)[:,1,:,:]
        dice_loss = self.seg_dice_loss(input_soft,target) # dice

        e_ce_loss = self.e_ce_loss(inputs, target, ellipse, ellipse_gradient)

        sub_loss = self.sub_loss(input_soft,target,ellipse)

        return ce_loss + dice_loss + e_ce_loss + sub_loss




# T Loss
class Sub_Neighbor_Loss(nn.Module):
    def __init__(self, ksize):
        super(Sub_Neighbor_Loss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')

        kk = int((ksize - 1) // 2)
        self.pad_r = nn.ReflectionPad2d(kk)
        self.max_pool2d = nn.MaxPool2d(kernel_size=ksize, stride=1, padding=0)
    def forward(self, input_soft, target, ellipse):
        n, h, w= target.size()
        target_clean = target.clone()
        target_clean[target_clean==255] = 0 # ignore 255
        input_s = torch.unsqueeze(input_soft, dim=1)
        target_f = (torch.unsqueeze(target_clean, dim=1)).float()

        neighbor_num = 4 

        input_gradient = input_s*neighbor_num
        target_gradient = target_f*neighbor_num
        
        if neighbor_num==4:# neighbor 4
            input_gradient[:,:,1:,:] -= input_s[:,:,:-1,:]  
            input_gradient[:,:,:-1,:] -= input_s[:,:,1:,:]  
            input_gradient[:,:,:,1:] -= input_s[:,:,:,:-1]  
            input_gradient[:,:,:,:-1] -= input_s[:,:,:,1:]   

            target_gradient[:,:,1:,:] -= target_f[:,:,:-1,:]  
            target_gradient[:,:,:-1,:] -= target_f[:,:,1:,:]  
            target_gradient[:,:,:,1:] -= target_f[:,:,:,:-1]  
            target_gradient[:,:,:,:-1] -= target_f[:,:,:,1:]  
        elif neighbor_num==8:# # neighbor 8
            input_gradient[:,:,1:,:] -= input_s[:,:,:-1,:]  
            input_gradient[:,:,:-1,:] -= input_s[:,:,1:,:]  
            input_gradient[:,:,:,1:] -= input_s[:,:,:,:-1]  
            input_gradient[:,:,:,:-1] -= input_s[:,:,:,1:]  

            input_gradient[:,:,1:,1:] -= input_s[:,:,:-1,:-1]  
            input_gradient[:,:,:-1,1:] -= input_s[:,:,1:,:-1]  
            input_gradient[:,:,1:,:-1] -= input_s[:,:,:-1,1:]   
            input_gradient[:,:,:-1,:-1] -= input_s[:,:,1:,1:]   

            target_gradient[:,:,1:,:] -= target_f[:,:,:-1,:]  
            target_gradient[:,:,:-1,:] -= target_f[:,:,1:,:]  
            target_gradient[:,:,:,1:] -= target_f[:,:,:,:-1] 
            target_gradient[:,:,:,:-1] -= target_f[:,:,:,1:]  

            target_gradient[:,:,1:,1:] -= target_f[:,:,:-1,:-1]  
            target_gradient[:,:,:-1,1:] -= target_f[:,:,1:,:-1]  
            target_gradient[:,:,1:,:-1] -= target_f[:,:,:-1,1:]  
            target_gradient[:,:,:-1,:-1] -= target_f[:,:,1:,1:]  

        target_diate = self._dilate_(target_f)
        target_erode = self._erode_(target_f)
        target_band = target_diate - target_erode
        target_band[target_band>0] = 1
        target_band[target_band<1] = 0
        return (self.mse(input_gradient[:,0,:,:],target_gradient[:,0,:,:]) * target_band).sum()/target_band.sum()

    def _dilate_(self, bin_img):
        bin_img = self.pad_r(bin_img)
        out = self.max_pool2d(bin_img)
        return out

    def _erode_(self, bin_img):
        out = 1 - self._dilate_(1 - bin_img)
        return out


# E Loss
class E_CrossEntropyLoss(nn.Module):
    def __init__(self, mode, weight=None):
        super(E_CrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, reduction='none',ignore_index=255).to(device)
        self.mode = mode
    def forward(self, input, target, ellipse, ellipse_gradient):
        # n, c, h, w= input.size()
        target = target.long()
        img_loss = self.criterion(input, target)
        if self.mode == 'whole':
            img_loss *= ellipse
            return torch.sum(img_loss) / ellipse.sum()
        elif self.mode == 'gradient': 

            ellipse_gradient = 1 - ellipse_gradient
            ellipse_gradient = ellipse_gradient * ellipse# inside ellipse
            img_loss = img_loss * ellipse_gradient.data 
            return torch.sum(img_loss) / ellipse_gradient.sum()
        else:
            raise Exception ("loss error")


#  Dice
class Seg_DiceLoss(nn.Module):
    def __init__(self):
        super(Seg_DiceLoss, self).__init__()
        pass
    def forward(self, input, targets):
        target_dice = targets.clone()
        target_dice[target_dice==255] = 0 # ignore 255
        N = target_dice.size()[0]# batchsize
        smooth = 1e-8 # smooth
        input_flat = input.view(N, -1)# reshape 
        targets_flat = target_dice.view(N, -1)
        intersection = input_flat * targets_flat 
        N_dice_eff = (2 * intersection.sum(1)+smooth) / (input_flat.sum(1) + targets_flat.sum(1)+smooth)
        loss = 1 - N_dice_eff.sum() / N # average a batch
        return loss

# cross entropy
class Seg_CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super(Seg_CrossEntropyLoss, self).__init__()
        # ignore background 255
        self.criterion = nn.CrossEntropyLoss(weight=weight, reduction='none',ignore_index=255).to(device)
    def forward(self, input, target):
        n, c, h, w= input.size()
        target = target.long()
        return torch.sum(self.criterion(input, target)) / ((target==1).sum()+(target==0).sum())


