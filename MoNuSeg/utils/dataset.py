# -*- coding=utf-8 -*-

import os
import numpy as np 
import torch 
import torch.utils.data as data
import pickle

from utils.transforms import Transforms 

def get_trainingset():
    # path of training data 
    trainingset_path = "MoNuSeg/MoNuSeg_2018_Training_Data.dat" #You may get this file at https://mcprl.com/html/dataset/NuSEA.html
    trainingset_file = open(trainingset_path,"rb")
    training_dict = pickle.load(trainingset_file) 
    trainingset_file.close()
    print(training_dict["info"],training_dict["average_size"])# The average dimension after extending each side by one-fifth of its length.
    # print(training_dict["samples"][0].keys())#dict_keys(['class_dir', 'image_name', 'position_w_h', 'crop_image', 'crop_truth', 'ellipse'])
    return training_dict

class DatasetForTraining(data.Dataset):
    def __init__(self, input_size=None):
        super(DatasetForTraining, self).__init__()
        training_dict = get_trainingset()
        self.filelist = training_dict["samples"]
        self.transforms = Transforms(input_size) 
    def __getitem__(self, index):
        input_data = self.filelist[index]["crop_image"]
        label_data = self.filelist[index]["crop_truth"]
        assist_ellipse = self.filelist[index]["ellipse"]
        assert input_data.shape[:-1]==label_data.shape,"crop_img and crop_truth mismatch"          
        sample = {'image': input_data, 'label': label_data, "ellipse": assist_ellipse}
        sample = self.transforms(sample) 
        return sample
    def __len__(self):
        return len(self.filelist)
