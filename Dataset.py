######################################
# Author : Wan-Cyuan Fan
######################################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import os.path
import sys
import string
import numpy as np
import pandas as pd
import random
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

desired_size = 224
num_limit_lower = 250
num_limit_up = 250

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

def add_gaussian_noise(img):
    mean = 0
    var = 10
    sigma = var ** 0.5
    (h, w) = img.shape[:2]
    gaussian = np.random.normal(mean, sigma, (h, w)) #  np.zeros((224, 224), np.float32)

    noisy_image = np.zeros(img.shape, np.float32)

    if len(img.shape) == 2:
        noisy_image = img + gaussian
    else:
        noisy_image[:, :, 0] = img[:, :, 0] + gaussian
        noisy_image[:, :, 1] = img[:, :, 1] + gaussian
        noisy_image[:, :, 2] = img[:, :, 2] + gaussian

    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image

def rotation(img):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    angle180 = 180
    M = cv2.getRotationMatrix2D(center, angle180, 1.0)
    rotated180 = cv2.warpAffine(img, M, (w, h))
    return rotated180

class Dataset(Dataset):
    def __init__(self, csv_path, mode = "not_test",sample = False, filename = False,argu = False):
        self.argu = argu
        self.mode = mode
        self.csv_path = csv_path
        self.filename = filename
        if self.mode == "test":
            self.img_names = os.listdir(csv_path)
            self.img_names.sort()
            self.len = len(self.img_names)
        else:
            self.filenames = list(pd.read_csv(csv_path).image_name)
            self.labels = list(pd.read_csv(csv_path).label)
            if sample == True:
                delete_list = []
                for idx , name in enumerate(self.filenames):
                    img_number = int(name.replace(".jpg","")[-5:])
                    if img_number >= num_limit_lower:
                        delete_list.append(idx)
                delete_list = delete_list[::-1]
                for i in delete_list:
                    del(self.filenames[i])
                    del(self.labels[i])

            self.len = len(self.filenames)

    def __getitem__(self,idx):

        if self.mode == "test":
            im = cv2.imread(os.path.join(self.csv_path,self.img_names[idx]))
            #############################################################
            # resize and padding
            #############################################################
            old_size = im.shape[:2]
            ratio = float(desired_size)/max(old_size)
            new_size = tuple([int(x*ratio) for x in old_size])

            im = cv2.resize(im, (new_size[1], new_size[0]))

            delta_w = desired_size - new_size[1]
            delta_h = desired_size - new_size[0]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)

            color = [0, 0, 0]
            new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
            if self.argu == True and random.random() <= 0.2:
                new_im = rotation(new_im)
                new_im = add_gaussian_noise(new_im)
            new_img = cv2.cvtColor(new_im, cv2.COLOR_BGR2RGB)
            # new_img = cv2.resize(new_img,(64,64))
            #############################################################
            new_img = np.array(new_img)
            new_img = transform(new_img)
            #new_img = torch.FloatTensor(np.array(new_img).transpose(2, 0, 1))
            if self.filename == True:
                return new_img ,self.img_names[idx]
            return new_img

        else:
            im = cv2.imread(os.path.join("dataset_public",self.filenames[idx]))
            #############################################################
            # resize and padding
            #############################################################
            old_size = im.shape[:2]
            ratio = float(desired_size)/max(old_size)
            new_size = tuple([int(x*ratio) for x in old_size])

            im = cv2.resize(im, (new_size[1], new_size[0]))

            delta_w = desired_size - new_size[1]
            delta_h = desired_size - new_size[0]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)

            color = [0, 0, 0]
            new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)

            new_img = cv2.cvtColor(new_im, cv2.COLOR_BGR2RGB)
            # new_img = cv2.resize(new_img,(64,64))
            #############################################################
            new_img = np.array(new_img)
            #print(new_img.shape)
            new_img = transform(new_img)
            #print(new_img.shape)
            #new_img = torch.FloatTensor(np.array(new_img).transpose(2, 0, 1))
            #print(new_img.shape)
            label = torch.LongTensor([int(self.labels[idx])])
            if self.filename == True:
                return new_img,label,self.filenames[idx]
            return new_img,label


    def __len__(self):
        return self.len

class Valid_Dataset(Dataset):
    def __init__(self, csv_path, mode = "train",sample = True):
        self.mode = mode
        self.csv_path = csv_path
        if self.mode == "test":
            self.img_names = os.listdir(csv_path)
            self.len = len(self.img_names)
        else:
            self.filenames = list(pd.read_csv(csv_path).image_name)
            self.labels = list(pd.read_csv(csv_path).label)
            if sample == True:
                delete_list = []
                for idx , name in enumerate(self.filenames):
                    img_number = int(name.replace(".jpg","")[-5:])
                    if img_number < num_limit_up:
                        delete_list.append(idx)
                delete_list = delete_list[::-1]
                for i in delete_list:
                    del(self.filenames[i])
                    del(self.labels[i])

            self.len = len(self.filenames)

    def __getitem__(self,idx):

        if self.mode == "test":
            im = cv2.imread(os.path.join(self.csv_path,self.img_names[idx]))
            #############################################################
            # resize and padding
            #############################################################
            old_size = im.shape[:2]
            ratio = float(desired_size)/max(old_size)
            new_size = tuple([int(x*ratio) for x in old_size])

            im = cv2.resize(im, (new_size[1], new_size[0]))

            delta_w = desired_size - new_size[1]
            delta_h = desired_size - new_size[0]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)

            color = [0, 0, 0]
            new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)

            new_img = cv2.cvtColor(new_im, cv2.COLOR_BGR2RGB)
            # new_img = cv2.resize(new_img,(64,64))
            #############################################################
            new_img = np.array(new_img)
            new_img = transform(new_img)
            #new_img = torch.FloatTensor(np.array(new_img).transpose(2, 0, 1))
            return new_img

        else:
            im = cv2.imread(os.path.join("dataset_public",self.filenames[idx]))
            #############################################################
            # resize and padding
            #############################################################
            old_size = im.shape[:2]
            ratio = float(desired_size)/max(old_size)
            new_size = tuple([int(x*ratio) for x in old_size])

            im = cv2.resize(im, (new_size[1], new_size[0]))

            delta_w = desired_size - new_size[1]
            delta_h = desired_size - new_size[0]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)

            color = [0, 0, 0]
            new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)

            new_img = cv2.cvtColor(new_im, cv2.COLOR_BGR2RGB)
            # new_img = cv2.resize(new_img,(64,64))
            #############################################################
            new_img = np.array(new_img)
            #print(new_img.shape)
            new_img = transform(new_img)
            #print(new_img.shape)
            #new_img = torch.FloatTensor(np.array(new_img).transpose(2, 0, 1))
            #print(new_img.shape)
            label = torch.LongTensor([int(self.labels[idx])])

            return new_img,label

    def __len__(self):
        return self.len

def test():
    inf_csv_path = ["./dataset_public/infograph/infograph_train.csv","./dataset_public/infograph/infograph_test.csv"]
    qdr_csv_path = ["./dataset_public/quickdraw/quickdraw_train.csv","./dataset_public/quickdraw/quickdraw_test.csv"]
    skt_csv_path = ["./dataset_public/sketch/sketch_train.csv","./dataset_public/sketch/sketch_test.csv"]
    rel_csv_path = ["./dataset_public/real/real_train.csv"]
    test_path = "./dataset_public/test"

    train_dataset = Dataset(csv_path = test_path , mode = "test",filename = True)
    train_loader = DataLoader(train_dataset,batch_size=1,shuffle=False,num_workers=0)
    train_iter = iter(train_loader)
    print(len(train_loader.dataset))
    print(len(train_loader))
    for epoch in range(1):
        #img,target,filename = next(train_iter)
        img,filename = next(train_iter)
        print("img shape : ",img.shape)
        #print("target shape : ",target.shape)
        print("filename :",filename)
        #img shape :  torch.Size([1, 7, 224, 224, 3])
        #target shape :  torch.Size([1])
        #print("target is : ",target[0])
        img = np.array(img[0,:,:,:]).transpose(1,2,0)
        plt.imshow(img)
        plt.savefig("./test.png")

if __name__ == "__main__":
    test()
