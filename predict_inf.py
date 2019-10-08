import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import skimage.io
import skimage
import os
import sys
import time
import pandas as pd
import random
import pickle
import Dataset_pred.Dataset_inf as Dataset
import model_64 as model
import torchvision.models as models
from scipy.spatial import distance
from numpy.linalg import inv

random.seed(312)
torch.manual_seed(312)

number_of_domain = 4

def save_checkpoint(checkpoint_path, model):#, optimizer):
    state = {'state_dict': model.state_dict()}
             #'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path,map_location = "cuda")
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)

def main():
    batch_size = 2
    batch_size_test = 2
    feature_size = 2048 * 1 * 1
    print("Check img size!!!!!!!!!!!!")
    # load my Dataset
    inf_csv_path = ["./dataset_public/infograph/infograph_train.csv","./dataset_public/infograph/infograph_test.csv"]
    qdr_csv_path = ["./dataset_public/quickdraw/quickdraw_train.csv","./dataset_public/quickdraw/quickdraw_test.csv"]
    skt_csv_path = ["./dataset_public/sketch/sketch_train.csv","./dataset_public/sketch/sketch_test.csv"]
    rel_csv_path = ["./dataset_public/real/real_train.csv"]
    test_path = "./dataset_public/test"

    test_dataset = Dataset.Dataset(csv_path = inf_csv_path[1],filename = True)

    test_loader = DataLoader(test_dataset ,batch_size = batch_size_test ,shuffle=False ,num_workers=1)

    print('the target dataset has %d size.' % (len(test_loader)))
    print('the batch_size is %d' % (batch_size))

    # Pre-train models
    modules = list(models.resnet152(pretrained = True).children())[:-1]
    encoder = nn.Sequential(*modules)
    #encoder = model.Encoder()
    classifier = model.Classifier(feature_size)

    # GPU enable
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        encoder = encoder.to(device)
        classifier = classifier.to(device)

    #loading models
    encoder_path = sys.argv[1]
    classifier_path = sys.argv[2]
    load_checkpoint(encoder_path,encoder)
    load_checkpoint(classifier_path,classifier)

    encoder.eval()
    classifier.eval()
    filename_save = []
    output_save = []
    total_acc = 0.
    with torch.no_grad():
        for index, (imgs,labels,filenames) in enumerate(test_loader):
            print("\r%d/%d"%(index+1,len(test_loader)),end="")
            output_list = []
            imgs = Variable(imgs).to(device)
            labels = Variable(labels.view(-1)).to(device)

            hidden = encoder(imgs)
            output = classifier(hidden)
            preds = output.argmax(1).cpu()
            acc = np.mean((preds.detach().cpu() == labels.cpu()).numpy())
            total_acc += acc
            for filename in filenames:
                filename_save.append(filename[-10:])
            for out in preds.detach().cpu():
                output_save.append(out)
        total_acc = total_acc/len(test_loader)
        print("Acc = ",total_acc)

    print(len(filename_save))
    print(len(output_save))
    # save csv
    file = open("./submission/pred_inf.csv","w")
    file.write("image_name,label\n")
    for i,(filename,data) in enumerate(zip(filename_save,output_save)):
        file.write("test/%s,%d\n"%(filename,int(data)))
    return 0


if __name__ == '__main__':
    main()
