from os import listdir
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
import glob
import os
import os.path
import sys
import string
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

limit_data_num = 150
desired_size = 128

############################################################
######      load data & labels (single directory)
############################################################
# -> data loading
def load_data(type,train_test_mode,csv_path,num_limit = limit_data_num):
    print("Note : num_limit = %d"%(num_limit))
    img_paths = np.array(pd.read_csv(csv_path)["image_name"])
    img_label = np.array(pd.read_csv(csv_path)["label"])

    X = []
    Y = []
    for i , (img_p , img_l) in enumerate(zip(img_paths,img_label)):
        img_number = int(img_p.replace(".jpg","")[-3:])
        if train_test_mode == "train" and img_number >= num_limit:
            continue
        print("\r%d/%d"%(i+1,len(img_paths)),end = "")
        im = cv2.imread(os.path.join("dataset_public",img_p))
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
        new_img = new_img / 255.
        label = int(img_l)

        # print(img_p," ",label)
        X.append(new_img)
        Y.append(label)

    X = torch.FloatTensor(np.array(X).transpose(0, 3, 1, 2))
    Y = torch.LongTensor(Y)

    print("")
    print("X shape : ",X.shape)
    print("Y shape : ",Y.shape)
    # test img save
    #img = np.array(X[0])
    #img = img.transpose(1, 2, 0)
    #print(img.shape)
    #im = plt.imshow(img)
    #plt.savefig("test.png")

    with open("./data/X_%s_%s.pkl"%(type,train_test_mode), "wb") as f:
        pickle.dump(X, f)
    with open("./data/Y_%s_%s.pkl"%(type,train_test_mode), "wb") as f:
        pickle.dump(Y, f)

    return 0

def load_test(path):
    img_names = os.listdir(path)

    X = []
    for i , img_name in enumerate(img_names):
        print("\r%d/%d"%(i+1,len(img_names)),end = "")
        img = cv2.imread(os.path.join(path,img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.
        img = cv2.resize(img,(64,64))
        X.append(img)


    X = np.array(X).transpose(0, 3, 1, 2)
    X = torch.FloatTensor(X)
    print("")
    print("X shape : ",X.shape)

    with open("./data/X_test_test.pkl", "wb") as f:
        pickle.dump(X, f)

    return 0

def main():
    type = ["infograph","quickdraw","sketch","real","test"]
    train_test_mode = ["train","test"]
    inf_csv_path = ["./dataset_public/infograph/infograph_train.csv","./dataset_public/infograph/infograph_test.csv"]
    qdr_csv_path = ["./dataset_public/quickdraw/quickdraw_train.csv","./dataset_public/quickdraw/quickdraw_test.csv"]
    skt_csv_path = ["./dataset_public/sketch/sketch_train.csv","./dataset_public/sketch/sketch_test.csv"]
    rel_csv_path = ["./dataset_public/real/real_train.csv"]
    test_path = "./dataset_public/test"


    print("<"+"="*40+">")
    print("Loading %s dataset..."%(type[0]))
    inf_train = load_data(type[0],train_test_mode[0],inf_csv_path[0])
    inf_test = load_data(type[0],train_test_mode[1],inf_csv_path[1])
    print("<"+"="*40+">")
    print("Loading %s dataset..."%(type[1]))
    qdr_train = load_data(type[1],train_test_mode[0],qdr_csv_path[0])
    qdr_test = load_data(type[1],train_test_mode[1],qdr_csv_path[1])
    print("<"+"="*40+">")
    print("Loading %s dataset..."%(type[2]))
    skt_train = load_data(type[2],train_test_mode[0],skt_csv_path[0])
    skt_test = load_data(type[2],train_test_mode[1],skt_csv_path[1])
    print("<"+"="*40+">")
    print("Loading %s dataset..."%(type[3]))
    rel_train = load_data(type[3],train_test_mode[0],rel_csv_path[0])

    print("<"+"="*40+">")
    print("Loading %s dataset..."%(type[4]))
    test_data = load_test(test_path)



if __name__ == "__main__":
    main()
