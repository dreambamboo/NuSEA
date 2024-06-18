# -*- coding=utf-8 -*-

import numpy as np
import torch
import time
from tqdm import *
import os
import torch.optim as optim
from torch.utils.data import DataLoader

from net.my_net import MyNet
from utils.dataset import DatasetForTraining 
from utils.loss import LossFunction 
from utils.visdom_display import Display 
from utils.log_write import Logger 
from utils.eval import EvaluationPixel,EvaluationInstance 
import pickle
import cv2
from PIL import Image

from apex import amp 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IsFloat16 = False 

class Option(object):
    def __init__(self):
        self.state = 'TEST'#'TRAIN' 
        self.is_assist = True 
        self.seed = 3407
        self.input_size = 80
        self.batchsize = 256
        self.model_dir = '../Models_with_Parameters'
        self.pretrained = True 
        self.pretrain_net = os.path.join(self.model_dir,'MoNuSeg.pth')#You may get this file at https://mcprl.com/html/dataset/NuSEA.html

        self.lr_decrease = True
        self.lr = 0.003
        self.iter_print = 2000
        self.optimizer_select = "RMSprop" 

        self.env = 'cell-MoNuSeg'
        self.net_select = 'unet-dcn'
        self.early_stop = 50
        self.epochs = 100 
        self.check_epoch = 5
        self.model_tag = 'test-'+self.net_select
        self.curve_tag = self.model_tag

        self.testset_crop_images = 'MoNuSegTestData.dat' #You may get this file at https://mcprl.com/html/dataset/NuSEA.html
        self.testset_truth = 'MoNuSegTestData-truth.dat' #You may get this file at https://mcprl.com/html/dataset/NuSEA.html
        self.test_threshold = 0.5 
        self.is_eval_instance = True 
        self.is_assist_test = False 
        self.is_save = False 
        self.save_results_dir = './results'


class Cell_Segmentation(object):
    def __init__(self,params):
        self.params = params 
        torch.manual_seed(self.params.seed)
        torch.cuda.manual_seed(self.params.seed)
        print ("{}Initializing{}".format('='*10,'='*10))
        self.net_loss = LossFunction() 
        self.eval_pixel = EvaluationPixel(self.params.is_save)
        self.eval_instance = EvaluationInstance(self.params.is_save)
        self.log = Logger(os.path.join(self.params.model_dir, 'logs', self.params.model_tag+'.log'),level='info')
        self.log.logger.info("\nState:{}\nBatchSize:{}\nPretrained:{}\nModel:{}".format(
            self.params.state,self.params.batchsize,self.params.pretrain_net,
            self.params.model_tag))
        if self.params.state == 'TRAIN':
            self.dataloader = DataLoader(dataset=DatasetForTraining(self.params.input_size), num_workers=4, batch_size=self.params.batchsize, shuffle=True)
        elif self.params.state == 'TEST':
            pass
        else:
            raise Exception ('Error: state error!(__init__)')
        self._load_mynet_()
        self.display_loss = Display(self.params.env,self.params.state,self.params.curve_tag)
        if self.params.optimizer_select == "SGD":
            self.optimizer = optim.SGD(self.mynet.parameters(), lr=self.params.lr) 
            if IsFloat16: # 混合精度
                self.mynet, self.optimizer = amp.initialize(self.mynet, self.optimizer, opt_level="O1") 
            if self.params.lr_decrease:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.params.epochs, 0.00001)
        elif self.params.optimizer_select == "RMSprop":
            self.optimizer = optim.RMSprop(self.mynet.parameters(), lr=self.params.lr, weight_decay=1e-8, momentum=0.9)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,'max', patience=2)
        elif self.params.optimizer_select == "Adam":
            self.optimizer = optim.Adam(self.mynet.parameters(), lr=self.params.lr) 
            if IsFloat16:
                self.mynet, self.optimizer = amp.initialize(self.mynet, self.optimizer, opt_level="O1") 
            if self.params.lr_decrease:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.params.epochs, 0.00001)
    def _load_mynet_(self, mode=None):
        if self.params.pretrained==True:
            self.mynet = MyNet(self.params.net_select,self.params.is_assist).to(device) 
            self.mynet.load_state_dict(torch.load(self.params.pretrain_net))
        else:
            self.mynet = MyNet(self.params.net_select,self.params.is_assist).to(device)   

    def _checkpoint_(self, str_, iter):
        model_out_path = os.path.join(self.params.model_dir,self.params.model_tag +'_'+ str_ + str(iter) + '.pth')
        torch.save(self.mynet.state_dict(), model_out_path)
        self.log.logger.info("====> Checkpoint saved to {}\n".format(model_out_path))
   
    def __call__(self):
        if self.params.state=='TRAIN':
            self._train_()
        elif self.params.state=='TEST':
            self._test_()
        else:
            raise Exception("CALL Error!")

    def _train_(self):

        for epoch in range(self.params.early_stop):
            self.mynet.train()
    
            epoch_loss = 0 
            dataloader_size = len(self.dataloader)

            t_iter = time.time() 
            t_epoch = time.time() 

            for iteration,sample in enumerate(self.dataloader):
                torch.cuda.empty_cache()
                inputs = sample['image'].to(device)
                labels = sample['label'].to(device)
                ellipse = sample['ellipse'].to(device)
                ellipse_gradient = sample['ellipse_gradient'].to(device)
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = self.mynet(inputs,ellipse) 
                    loss = self.net_loss(outputs,labels,ellipse,ellipse_gradient) 
                    if IsFloat16:
                        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    self.optimizer.step()

                epoch_loss += float(loss.item() )

                self.display_loss('TRAIN',iteration+1+dataloader_size*epoch,loss.item())
                
                if (iteration+1) % self.params.iter_print == 0:
                    print("==> Epoch[{}]({}/{}): lr: {} Loss: {:.4f} Time: {:4f}".\
                        format((epoch+1), iteration+1, dataloader_size,self.optimizer.param_groups[0]['lr'], loss.item(), time.time()-t_iter))
                    t_iter = time.time()
                

            epoch_loss /= dataloader_size 
            self.log.logger.info("{}\n==> Epoch {} Complete: Avg. Loss: {:.4f} Time:{:.4f}".format('_'*10,epoch+1, epoch_loss, time.time()-t_epoch))

            if self.params.lr_decrease==True:
                if self.params.optimizer_select == "RMSprop":
                    self.scheduler.step(metrics=self._test_()) 
                else:
                    self.scheduler.step() 
            if (epoch+1) % self.params.check_epoch == 0:
                self._checkpoint_('Epoch',epoch+1)
                if self.params.optimizer_select == "RMSprop":
                    pass 
                else:
                    self._test_() 
            
    def _test_(self):
        test_time = time.time()
        testset_images_file = open(self.params.testset_crop_images,'rb')
        testset_dict = pickle.load(testset_images_file)["samples"]
        testset_images_file.close()
        testset_truth_file = open(self.params.testset_truth, 'rb')
        testset_truth = pickle.load(testset_truth_file)["samples"]
        testset_truth_file.close()

        eval_count = 0 
        eval_pixel_metrics = {"dice":0,
                        "iou":0,
                        "acc":0,
                        "recall":0,
                        "precision":0}
        eval_instance_metrics = {"DQ-F1":0,
                "SQ":0,
                "PQ":0,
                "AJI":0,
                "dice_obj":0,
                "iou_obj":0,
                "hausdorff_obj":0,
                "Average_Symmetric_Surface_Distance":0,
                "Maximum_mean_surface_distance":0,
                }

        for class_dir in ['none']:
            for img_name in tqdm(testset_dict['none'].keys()):

                cell_predict_list = np.zeros((1,self.params.input_size,self.params.input_size)) 
                cell_ellipse_list = [] 

                batch_img = np.zeros((1,self.params.input_size,self.params.input_size,3))
                batch_ellipse = np.zeros((1,self.params.input_size,self.params.input_size))

                for cell in testset_dict[class_dir][img_name]:
                    crop_image = cell["crop_image"] 
                    cx,cy,a,b,angle = cell["ellipse"]

                    label3 = np.zeros(crop_image.shape)
                    cv2.ellipse(label3, (np.int32(cx), np.int32(cy)),(np.int32(a/2), np.int32(b/2)), angle, 0, 360, (255,0,0),-1) 
                    ellipse = label3[:,:,0] 
                    cell_ellipse_list.append(ellipse)

                    crop_image, ellipse_resize = self._test_resize_(crop_image,ellipse)

                    crop_image = crop_image[np.newaxis,:,:,:] # 1*x*x*3 
                    ellipse_resize = ellipse_resize[np.newaxis,:,:] # 1*x*x
                    batch_img = np.concatenate((batch_img, crop_image), axis=0) # (N+1)*x*x*3 
                    batch_ellipse = np.concatenate((batch_ellipse, ellipse_resize), axis=0) # (N+1)*x*x

                    if batch_img.shape[0] == self.params.batchsize:
                        batch_output = self._test_single_batch_(batch_img[1:], batch_ellipse[1:]) # N*x*x  
                        cell_predict_list = np.concatenate((cell_predict_list, batch_output), axis=0) 
                        batch_img = np.zeros((1,self.params.input_size,self.params.input_size,3)) 
                        batch_ellipse = np.zeros((1,self.params.input_size,self.params.input_size)) 
                    else:
                        continue

                if batch_img.shape[0]>1: #
                    batch_output = self._test_single_batch_(batch_img[1:], batch_ellipse[1:]) # N*x*x  
                    cell_predict_list = np.concatenate((cell_predict_list, batch_output), axis=0) 
                else:
                    pass 

                cell_predict_list = cell_predict_list[1:] 
                truth = testset_truth[class_dir][img_name] 
                predict = np.zeros(truth.shape) #1000*1000*10

                pre_fill_index = 0 
                for cell_index, single_cell in enumerate(testset_dict[class_dir][img_name]):
                    cell_predict = cell_predict_list[cell_index] 
                    cell_h, cell_w, c = single_cell["crop_image"].shape 
                    cell_ellipse = cell_ellipse_list[cell_index] 
                    cell_ellipse[cell_ellipse>0] = 1 
                    cell_predict_resize = self._test_resize_trans_(cell_predict, cell_w, cell_h)

                    if self.params.is_assist_test:
                        # cell_predict_ellipse_mask = cell_ellipse*255 # human（baseline）
                        cell_predict_ellipse_mask = cell_ellipse*cell_predict_resize # only information inside ellipses
                    else:
                        cell_predict_ellipse_mask = cell_predict_resize # test without ellipse assistance
                    cell_predict_ellipse_mask = cell_predict_ellipse_mask/255
                    cell_predict_ellipse_mask[cell_predict_ellipse_mask>self.params.test_threshold] = 1
                    cell_predict_ellipse_mask[cell_predict_ellipse_mask<1] = 0 
                    position_w, position_h = single_cell["position_w_h"] # crop

                    if cell_predict_ellipse_mask.sum()==0:
                        # nothing predicted
                        continue
                    else:
                        pre_fill_index += 1
                        
                    pre_last0 = predict[position_h:position_h+cell_h,position_w:position_w+cell_w,0]
                    pre_intersection = pre_last0*cell_predict_ellipse_mask #
                    if pre_intersection.sum()==0: 
                        predict[position_h:position_h+cell_h,position_w:position_w+cell_w,0] += cell_predict_ellipse_mask*pre_fill_index 
                    else: 
                        pre_intersection[pre_intersection>0] = 1
                        predict[position_h:position_h+cell_h,position_w:position_w+cell_w,0] += (cell_predict_ellipse_mask*(1-pre_intersection)*pre_fill_index) 
                        for pre_layer in range(1,10): 
                            if (predict[position_h:position_h+cell_h,position_w:position_w+cell_w,pre_layer]*pre_intersection).sum()>0:
                                continue 
                            else: 
                                predict[position_h:position_h+cell_h,position_w:position_w+cell_w,pre_layer] += pre_intersection*pre_fill_index
                                break

                eval_count += 1
                eval_pixel_dict = self.eval_pixel(predict.astype(np.float32),truth.astype(np.float32),os.path.join(self.params.save_results_dir,class_dir+'_'+img_name)) 
                for metric in eval_pixel_metrics.keys():
                    eval_pixel_metrics[metric] += eval_pixel_dict[metric]

                if self.params.is_eval_instance:
                    eval_instance_dict = self.eval_instance(predict.astype(np.float32),truth.astype(np.float32),os.path.join(self.params.save_results_dir,class_dir+'_'+img_name))
                    for metric in eval_instance_metrics.keys():
                        eval_instance_metrics[metric] += eval_instance_dict[metric]
        self.log.logger.info("====> Test Time: {} ".format(time.time()-test_time))
        for metric in eval_pixel_metrics.keys():
            eval_pixel_metrics[metric] /= eval_count
            self.log.logger.info("\t{}:{:.4f}".format(metric,eval_pixel_metrics[metric]))
        if self.params.is_eval_instance:
            for metric in eval_instance_dict.keys():
                eval_instance_metrics[metric] /= eval_count
                self.log.logger.info("\t{}:{:.4f}".format(metric,eval_instance_metrics[metric]))

        if self.params.optimizer_select == 'RMSprop':
            return eval_pixel_metrics["dice"]

    def _test_single_batch_(self, batch_img, batch_ellipse):

        self.mynet.eval()
        batch_img, batch_ellipse = self._test_normalize_(batch_img, batch_ellipse)
        with torch.no_grad():
            output = self.mynet(batch_img, batch_ellipse) # N*2*x*x
            output = torch.softmax(output, dim=1)   
        output = output.cpu().data.numpy()
        return output[:,1,:,:]   

    def _test_normalize_(self, batch_img, batch_ellipse):
        batch_img = np.array(batch_img).astype(np.float16)
        batch_img = batch_img.transpose((0, 3, 1, 2))
        batch_img = torch.from_numpy(batch_img).float().to(device)
        batch_ellipse = torch.from_numpy(batch_ellipse).float().to(device)
        return batch_img, batch_ellipse
    def _test_resize_(self,img,ellipse):
        h,w  = ellipse.shape
        resize_factor = self.params.input_size/max([h,w]) 
        h_resize = int(min([self.params.input_size, h*resize_factor]))
        w_resize = int(min([self.params.input_size, w*resize_factor]))

        img = np.array(Image.fromarray(img).resize((w_resize,h_resize),Image.ANTIALIAS).convert("RGB")).astype(np.float32)
        ellipse = np.array(Image.fromarray(ellipse).resize((w_resize,h_resize),Image.NEAREST).convert("L")).astype(np.float32)

        img_new = np.zeros((self.params.input_size,self.params.input_size,3)).astype(np.float32) 
        ellipse_new = np.zeros((self.params.input_size,self.params.input_size)).astype(np.float32)
        h_pos = max([int((self.params.input_size - h_resize)/2),0])
        w_pos = max([int((self.params.input_size - w_resize)/2),0])

        img_new[h_pos:h_pos+h_resize,w_pos:w_pos+w_resize,:] = img 
        ellipse_new[h_pos:h_pos+h_resize,w_pos:w_pos+w_resize] = ellipse
        return img_new, ellipse_new
    def _test_resize_trans_(self, img, w, h):       
        resize_factor = self.params.input_size/max([h,w]) 
        h_resize = int(min([self.params.input_size, h*resize_factor]))
        w_resize = int(min([self.params.input_size, w*resize_factor]))
        h_pos = max([int((self.params.input_size - h_resize)/2),0])
        w_pos = max([int((self.params.input_size - w_resize)/2),0])
        img_pre_original = img[h_pos:h_pos+h_resize,w_pos:w_pos+w_resize]
        img_pre_resize = np.array(Image.fromarray(img_pre_original*255).resize((w,h),Image.ANTIALIAS).convert("L")).astype(np.float32)
        return img_pre_resize

if __name__ == "__main__":
    opt = Option()
    obj = Cell_Segmentation(opt)
    obj()

