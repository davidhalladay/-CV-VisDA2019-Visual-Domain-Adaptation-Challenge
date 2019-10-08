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
import torchvision.models as models
import skimage.io
import skimage
import os
import time
import pandas as pd
import random
import pickle
import model_64 as model
import Dataset

random.seed(312)
torch.manual_seed(312)

def save_checkpoint(checkpoint_path, model):#, optimizer):
    state = {'state_dict': model.state_dict()}
             #'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def main():
    # parameters

    learning_rate = 0.001
    num_epochs = 50
    batch_size_train = 200
    batch_size_test = 100
    feature_size = 512*1*1

    # load my Dataset
    inf_csv_path = ["./dataset_public/infograph/infograph_train.csv","./dataset_public/infograph/infograph_test.csv"]
    qdr_csv_path = ["./dataset_public/quickdraw/quickdraw_train.csv","./dataset_public/quickdraw/quickdraw_test.csv"]
    skt_csv_path = ["./dataset_public/sketch/sketch_train.csv","./dataset_public/sketch/sketch_test.csv"]
    rel_csv_path = ["./dataset_public/real/real_train.csv"]
    test_path = "./dataset_public/test"
    source = rel_csv_path[0]
    test = rel_csv_path[0]
    print("source from : ",source)
    print("test from : ",test)
    inf_train_dataset = Dataset.Dataset(csv_path = source,sample = True)
    s_train_loader = DataLoader(inf_train_dataset ,batch_size = batch_size_train ,shuffle=True ,num_workers=1)
    test_dataset = Dataset.Valid_Dataset(csv_path = test)
    #test_dataset = Dataset.Dataset(csv_path = test)
    test_loader = DataLoader(test_dataset ,batch_size = batch_size_test ,shuffle=True ,num_workers=1)

    print('the source dataset has %d size.' % (len(inf_train_dataset)))
    print('the target dataset has %d size.' % (len(test_dataset)))
    print('the batch_size is %d' % (batch_size_train))

    # models setting
    modules = list(models.resnet18(pretrained = True).children())[:-1]
    feature_extractor = nn.Sequential(*modules)
    #feature_extractor = model.Encoder()
    label_predictor = model.Classifier(feature_size)


    # GPU enable
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        feature_extractor = feature_extractor.to(device)
        label_predictor = label_predictor.to(device)

    # setup optimizer
    optimizer = optim.Adam([{'params': label_predictor.parameters()},
                            {'params': feature_extractor.parameters()}], lr= learning_rate)

    #Lossfunction
    L_criterion = nn.NLLLoss()

    print("Starting training...")
    best_acc = 0.
    for epoch in range(num_epochs):
        feature_extractor.train()
        label_predictor.train()

        print("Epoch:", epoch+1)
        len_dataloader = len(s_train_loader)

        epoch_L_loss = 0.0

        if (epoch+1) == 8:
            for optimizer_t in optimizer.param_groups:
                optimizer_t['lr'] /= 2
        #    F_optimizer.param_groups[0]['lr'] /= 2
        #    L_optimizer.param_groups[0]['lr'] /= 2
        #    D_optimizer.param_groups[0]['lr'] /= 2

        if (epoch+1) == 15:
            for optimizer_t in optimizer.param_groups:
                optimizer_t['lr'] /= 2

        if (epoch+1) == 20:
            for optimizer_t in optimizer.param_groups:
                optimizer_t['lr'] /= 2

        if (epoch+1) == 25:
            for optimizer_t in optimizer.param_groups:
                optimizer_t['lr'] /= 2

        if (epoch+1) == 30:
            for optimizer_t in optimizer.param_groups:
                optimizer_t['lr'] /= 2
        #    F_optimizer.param_groups[0]['lr'] /= 2
        #    L_optimizer.param_groups[0]['lr'] /= 2
        #    D_optimizer.param_groups[0]['lr'] /= 2

        for i, source_data in enumerate(s_train_loader):
            source_img, source_label = source_data
            source_img = Variable(source_img).to(device)
            source_label = Variable(source_label).to(device)

            # train the feature_extractor
            optimizer.zero_grad()
            #optimizer_label.zero_grad()

            source_feature = feature_extractor(source_img)

            # Label_Predictor network
            src_label_output = label_predictor(source_feature)

            _, src_pred_arg = torch.max(src_label_output,1)
            src_acc = np.mean(np.array(src_pred_arg.cpu()) == np.array(source_label.view(-1).cpu()))
            loss = L_criterion(src_label_output, source_label.view(-1))

            epoch_L_loss += loss.item()

            loss.backward()
            #optimizer_label.step()
            optimizer.step()

            if (i % 20 == 0):
                print('Epoch [%d/%d], Iter [%d/%d] loss %.4f , LR = %.6f , Acc = %.4f'
                %(epoch+1, num_epochs, i+1, len_dataloader, loss.item(), optimizer.param_groups[0]['lr'],src_acc))

        # epoch done
        print('-'*80)

        feature_extractor.eval()
        label_predictor.eval()
        total_acc = 0.
        for i, test_data in enumerate(test_loader):
            imgs , labels = test_data
            imgs = Variable(imgs).to(device)
            labels = Variable(labels).to(device)
            feature = feature_extractor(imgs)

            # Label_Predictor network
            output = label_predictor(feature)
            _, pred_arg = torch.max(output,1)
            acc = np.mean(np.array(pred_arg.cpu()) == np.array(labels.view(-1).cpu()))
            if i % 100 == 0:
                print(acc)
            total_acc += acc
        total_acc = total_acc/len(test_loader)
        if total_acc > best_acc:
            best_acc = total_acc
            print("Best accuracy : ",best_acc)

    print("<"+"="*40+">")
    print("Best accuracy : ",best_acc)

    # shuffle
if __name__ == '__main__':
    main()
