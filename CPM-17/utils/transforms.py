# -*- coding=utf-8 -*-


import numpy as np 
import torchvision.transforms.functional as TF
import random
import torch
from PIL import Image
import cv2

Mean_RGB = [0,0,0]
Std_RGB = [1,1,1]


class Transforms(object):
    def __init__(self,input_size,mode='train'):
        self.Mean_RGB = np.array(Mean_RGB)
        self.Std_RGB = np.array(Std_RGB)
        self.input_size = input_size
        self.mode=mode
    def __call__(self, sample):
        if self.mode =='train':
            return self._train_(sample)
        else:
            return self._test_(sample)
    def _train_(self, sample):
        sample = self._normalize_(sample, self.input_size)
        return sample
    def _normalize_(self, sample, input_size):
        img = sample['image']
        mask = sample['label']
        h,w  = mask.shape
        resize_factor = input_size/max([h,w]) 
        h_resize = int(min([input_size, h*resize_factor]))
        w_resize = int(min([input_size, w*resize_factor]))
        cx,cy,a,b,angle = sample['ellipse']
        ellipse_gradient = self._ellipse_line_gradient_(cx,cy,a,b,angle,h,w) # line
        ellipse_gradient = Image.fromarray(ellipse_gradient).resize((w_resize,h_resize),Image.ANTIALIAS).convert("L")
        ellipse_gradient = np.array(ellipse_gradient).astype(np.float32)
        truth3 = np.zeros(img.shape)
        cv2.ellipse(truth3, (np.int32(cx), np.int32(cy)),(np.int32(a/2), np.int32(b/2)), angle, 0, 360, (255,0,0),-1) 
        ellipse = truth3[:,:,0]
        ellipse = Image.fromarray(ellipse).resize((w_resize,h_resize),Image.NEAREST).convert("L")
        ellipse = np.array(ellipse).astype(np.float32)
        img = Image.fromarray(img).resize((w_resize,h_resize),Image.ANTIALIAS).convert("RGB")
        mask = Image.fromarray(mask).resize((w_resize,h_resize),Image.NEAREST).convert("L")
        mask = np.array(mask).astype(np.int64)

        # img, mask, ellipse = self._random_flip_(img, mask, ellipse)
        # img, mask, ellipse = self._random_rotate_(img, mask, ellipse)
        img = self._random_color_(img)
        # img = self._random_noise_(np.array(img).astype(np.float32)) 
        img = np.array(img).astype(np.float32)

        img -= self.Mean_RGB
        img /= self.Std_RGB

        h_pos = max([int((input_size - h_resize)/2),0])
        w_pos = max([int((input_size - w_resize)/2),0])

        img_new = np.zeros((input_size,input_size,3)).astype(np.float32) 
        truth_new = (np.ones((input_size,input_size)).astype(np.int64))*255 
        ellipse_new = np.zeros((input_size,input_size)).astype(np.float32)
        ellipse_gradient_new = np.zeros((input_size,input_size)).astype(np.float32)

        img_new[h_pos:h_pos+h_resize,w_pos:w_pos+w_resize,:] = img 
        truth_new[h_pos:h_pos+h_resize,w_pos:w_pos+w_resize] = mask # 0,1,255
        ellipse_new[h_pos:h_pos+h_resize,w_pos:w_pos+w_resize] = ellipse
        ellipse_gradient_new[h_pos:h_pos+h_resize,w_pos:w_pos+w_resize] = ellipse_gradient

        img_new = img_new.transpose((2, 0, 1))
        img_new = torch.from_numpy(img_new).float()


        return {'image': img_new,
                'label':truth_new, 
                'ellipse':ellipse_new, 
                'ellipse_gradient':ellipse_gradient_new} 
    def _test_(self,sample):

        input_size = self.input_size
        img = sample['image']
        mask = sample['label']
        h,w  = mask.shape
        resize_factor = input_size/max([h,w]) 
        h_resize = int(min([input_size, h*resize_factor]))
        w_resize = int(min([input_size, w*resize_factor]))
        cx,cy,a,b,angle = sample['ellipse']
        ellipse_gradient = self._ellipse_line_gradient_(cx,cy,a,b,angle,h,w) 
        ellipse_gradient = Image.fromarray(ellipse_gradient).resize((w_resize,h_resize),Image.ANTIALIAS).convert("L")
        ellipse_gradient = np.array(ellipse_gradient).astype(np.float32)
        truth3 = np.zeros(img.shape)
        cv2.ellipse(truth3, (np.int32(cx), np.int32(cy)),(np.int32(a/2), np.int32(b/2)), angle, 0, 360, (255,0,0),-1) 
        ellipse = truth3[:,:,0]
        ellipse = Image.fromarray(ellipse).resize((w_resize,h_resize),Image.NEAREST).convert("L")
        ellipse = np.array(ellipse).astype(np.float32)
        img = Image.fromarray(img).resize((w_resize,h_resize),Image.ANTIALIAS).convert("RGB")
        mask = Image.fromarray(mask).resize((w_resize,h_resize),Image.NEAREST).convert("L")
        mask = np.array(mask).astype(np.int64)
        img = np.array(img).astype(np.float32)

        img -= self.Mean_RGB
        img /= self.Std_RGB

        h_pos = max([int((input_size - h_resize)/2),0])
        w_pos = max([int((input_size - w_resize)/2),0])

        img_new = np.zeros((input_size,input_size,3)).astype(np.float32) 
        truth_new = (np.ones((input_size,input_size)).astype(np.int64))*255 
        ellipse_new = np.zeros((input_size,input_size)).astype(np.float32)
        ellipse_gradient_new = np.zeros((input_size,input_size)).astype(np.float32)

        img_new[h_pos:h_pos+h_resize,w_pos:w_pos+w_resize,:] = img 
        truth_new[h_pos:h_pos+h_resize,w_pos:w_pos+w_resize] = mask # 0,1,255
        ellipse_new[h_pos:h_pos+h_resize,w_pos:w_pos+w_resize] = ellipse
        ellipse_gradient_new[h_pos:h_pos+h_resize,w_pos:w_pos+w_resize] = ellipse_gradient

        img_new = img_new.transpose((2, 0, 1))
        img_new = torch.from_numpy(img_new).float()


        return {'image': img_new,
                'label':truth_new, 
                'ellipse':ellipse_new, 
                'ellipse_gradient':ellipse_gradient_new} 
                


    def _ellipse_line_gradient_(self,cx,cy,a,b,angle,h,w):
        large_size = 150 
        X = np.arange(0,2*large_size)
        Y = np.arange(0,2*large_size)
        X,Y = np.meshgrid(X,Y)
        X = X-large_size 
        Y = Y-large_size

        aa = np.ceil(b)/2
        bb = np.ceil(a)/2
        if aa==bb: 
            R = np.sqrt(X**2+Y**2)/aa  
            R = 1-R
            R = R.astype(np.uint8)
            R[R<0] = 0 
        elif aa>bb:  
            cc = np.sqrt(aa**2-bb**2)
            R = (2*aa/(np.sqrt((Y+cc)**2+X**2)+np.sqrt((Y-cc)**2+X**2)) -1)/max([(aa/cc-1),1e-8])
            R[R<0] = 0 
            R = Image.fromarray(R*255).rotate(-angle, resample=Image.BILINEAR, expand=None, center=None, translate=None, fillcolor=None)
            R = np.array(R).astype(np.uint8)
        else: 
            tt = aa 
            aa = bb 
            bb = tt 
            cc = np.sqrt(aa**2-bb**2)
            R = (2*aa/(np.sqrt((X+cc)**2+Y**2)+np.sqrt((X-cc)**2+Y**2)) -1)/max([(aa/cc-1),1e-8])
            R[R<0] = 0 
            R = Image.fromarray(R*255).rotate(-angle, resample=Image.BILINEAR, expand=None, center=None, translate=None, fillcolor=None)
            R = np.array(R).astype(np.uint8)

        R = R[int(np.ceil(large_size-cy)):int(np.ceil(large_size-cy+h)),int(np.ceil(large_size-cx)):int(np.ceil(large_size-cx+w))] 
        return R

    def _ellipse_jittor_(self, ellipse_para):
        cx,cy,a,b,angle = ellipse_para
        a_ex = {0:a, 1:a+random.randint(0,1)}[random.randint(0,1)]
        b_ex = {0:b, 1:b+random.randint(0,1)}[random.randint(0,1)]
        cx_j = {0:cx,1:cx+random.randint(-1,1)}[random.randint(0,1)]
        cy_j = {0:cy,1:cy+random.randint(-1,1)}[random.randint(0,1)]
        angle_r = {0:angle,1:angle+random.randint(-1,1)}[random.randint(0,1)]

        return cx_j,cy_j,a_ex,b_ex,angle_r


    def _random_noise_(self, img, u=0, a=0.1):

        temp = random.randint(0, 1)
        h, w, c = img.shape
        noise = np.random.normal(u, a, (h, w, c)).astype(np.float32)
        img = {0: img + noise, 1: img}[temp]
        return img

    def _random_color_(self, img):
        img = TF.adjust_brightness(img, random.uniform(0.8,1.2))
        img = TF.adjust_contrast(img, random.uniform(0.8,1.2))
        img = TF.adjust_hue(img, random.uniform(-0.05,0.05))
        img = TF.adjust_saturation(img, random.uniform(0.8,1.2))
        return img

    def _random_flip_(self, img, mask, ellipse):
        temp = random.randint(0, 2)
        # img = {0:img, 1:TF.hflip(img), 2:TF.vflip(img)}[temp]
        # mask = {0:mask, 1:TF.hflip(mask), 2:TF.vflip(mask)}[temp]
        # ellipse = {0:ellipse, 1:TF.hflip(ellipse), 2:TF.vflip(ellipse)}[temp]
        img = {0:img, 1:img.transpose(Image.FLIP_LEFT_RIGHT), 2:img.transpose(Image.FLIP_TOP_BOTTOM)}[temp]
        mask = {0:mask, 1:mask.transpose(Image.FLIP_LEFT_RIGHT), 2:mask.transpose(Image.FLIP_TOP_BOTTOM)}[temp]
        ellipse = {0:ellipse, 1:ellipse.transpose(Image.FLIP_LEFT_RIGHT), 2:ellipse.transpose(Image.FLIP_TOP_BOTTOM)}[temp]
        
        return img, mask, ellipse

    def _random_rotate_(self, img, mask, ellipse, range_degree=45):
        if random.randint(0, 1):
            rotate_degree = random.uniform(-range_degree,range_degree)
            img = img.rotate(rotate_degree, Image.BILINEAR)
            mask = mask.rotate(rotate_degree, Image.NEAREST)
            ellipse = ellipse.rotate(rotate_degree, Image.NEAREST)
            return img, mask, ellipse
        else:
            return img, mask, ellipse

    
