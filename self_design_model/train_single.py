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
import time
import pandas as pd
import random
import pickle
import Dataset
from model.Classifier import Classifier
from model.Classifier import Domain_classifier
from model.Encoder import Encoder

random.seed(312)
torch.manual_seed(312)

number_of_domain = 4

def save_checkpoint(checkpoint_path, model):#, optimizer):
    state = {'state_dict': model.state_dict()}
             #'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def main():
    # parameters
    learning_rate = 0.01
    num_epochs = 50
    batch_size = 250
    feature_size = 2048
    test_index = 2

    # create the save log file
    print("Create the directory")
    if not os.path.exists("./save"):
        os.makedirs("./save")
    if not os.path.exists("./logfile"):
        os.makedirs("./logfile")
    if not os.path.exists("./logfile/MTL"):
        os.makedirs("./logfile/MTL")

    # load my Dataset
    type = ["infograph","quickdraw","sketch","real","test"]
    print("training set : %s ,%s, %s"%(type[0],type[1],type[3]))
    print("testing set : %s"%(type[2]))
    inf_train_dataset = Dataset.Dataset(mode = "train",type = type[0])
    inf_train_loader = DataLoader(inf_train_dataset ,batch_size = batch_size ,shuffle=True ,num_workers=1)
    qdr_train_dataset = Dataset.Dataset(mode = "train",type = type[1])
    qdr_train_loader = DataLoader(qdr_train_dataset ,batch_size = batch_size ,shuffle=True ,num_workers=1)
    skt_train_dataset = Dataset.Dataset(mode = "train",type = type[2])
    skt_train_loader = DataLoader(skt_train_dataset ,batch_size = batch_size ,shuffle=True ,num_workers=1)
    rel_train_dataset = Dataset.Dataset(mode = "train",type = type[3])
    rel_train_loader = DataLoader(rel_train_dataset ,batch_size = batch_size ,shuffle=True ,num_workers=1)

    test_dataset = Dataset.Dataset(mode = "test",type = type[0])
    test_loader = DataLoader(test_dataset ,batch_size = batch_size ,shuffle=True ,num_workers=1)

    print('the source dataset has %d size.' % (len(inf_train_dataset)))
    print('the target dataset has %d size.' % (len(test_dataset)))
    print('the batch_size is %d' % (batch_size))

    # Pre-train models
    encoder = Encoder()
    classifier = Classifier(feature_size)
    domain_classifier = Domain_classifier(feature_size,number_of_domain)

    # GPU enable
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        encoder = encoder.to(device)
        domain_classifier = domain_classifier.to(device)
        classifier = classifier.to(device)

    # setup optimizer
    optimizer_encoder = optim.Adam(encoder.parameters(), weight_decay = 1e-4 , lr= learning_rate)
    optimizer_domain_classifier = optim.Adam(domain_classifier.parameters(), weight_decay = 1e-4 , lr= learning_rate)
    optimizer_classifier = optim.Adam(classifier.parameters(), weight_decay = 1e-4 , lr= learning_rate)


    print("Starting training...")

    for epoch in range(num_epochs):
        print("Epoch:", epoch+1)

        train_loader = [inf_train_loader,qdr_train_loader,skt_train_loader,rel_train_loader]
        domain_labels = torch.LongTensor([[0 for i in range(batch_size)],[1 for i in range(batch_size)],[2 for i in range(batch_size)],[3 for i in range(batch_size)]])

        mtl_criterion = nn.CrossEntropyLoss()
        moe_criterion = nn.CrossEntropyLoss()

        encoder.train()
        domain_classifier.train()
        classifier.train()

        epoch_D_loss = 0.0
        epoch_C_loss = 0.0
        sum_trg_acc = 0.0
        sum_label_acc = 0.0
        sum_test_acc = 0.0

        for index, (inf,qdr,skt,rel,test) in enumerate(zip(train_loader[0],train_loader[1],train_loader[2],train_loader[3],test_loader)):

            optimizer_encoder.zero_grad()
            optimizer_classifier.zero_grad()
            optimizer_domain_classifier.zero_grad()

            # colculate the lambda_
            p = (index + len(train_loader[0]) * epoch)/(len(train_loader[0]) * num_epochs)
            lambda_ = 2.0 / (1. + np.exp(-10 * p)) - 1.0

            s1_imgs , s1_labels = inf
            s2_imgs , s2_labels = qdr
            s3_imgs , s3_labels = rel
            t1_imgs , _ = skt
            s1_imgs = Variable(s1_imgs).to(device) ; s1_labels = Variable(s1_labels).to(device)
            s2_imgs = Variable(s2_imgs).to(device) ; s2_labels = Variable(s2_labels).to(device)
            s3_imgs = Variable(s3_imgs).to(device) ; s3_labels = Variable(s3_labels).to(device)
            t1_imgs = Variable(t1_imgs).to(device)

            s1_feature = encoder(s1_imgs)
            #t1_feature = encoder(t1_imgs)

            # Testing
            test_imgs , test_labels = test
            test_imgs = Variable(test_imgs).to(device) ; test_labels = Variable(test_labels).to(device)
            test_feature = encoder(test_imgs)
            test_output = classifier(test_feature)
            test_preds = test_output.argmax(1).cpu()
            test_acc = np.mean((test_preds == test_labels.cpu()).numpy())

            # Classifier network
            s1_output = classifier(s1_feature)


            s1_preds = s1_output.argmax(1).cpu()

            s1_acc = np.mean((s1_preds == s1_labels.cpu()).numpy())
            s1_c_loss = mtl_criterion(s1_output,s1_labels)
            C_loss = s1_c_loss

            # Domain_classifier network with source domain
            #domain_labels = Variable(domain_labels).to(device)
            #s1_domain_output = domain_classifier(s1_feature,lambda_)

            #s1_domain_preds = s1_domain_output.argmax(1).cpu()
            #if index == 10:
            #    print(s1_domain_preds)
            #s1_domain_acc = np.mean((s1_domain_preds == 0).numpy())
            #print(s1_domain_output.shape)
            #print(s1_domain_output[0])
            #s1_d_loss = moe_criterion(s1_domain_output,domain_labels[0])
            #D_loss_src = s1_d_loss
            #print(D_loss_src.item())

            # Domain_classifier network with target domain
            #t1_domain_output = domain_classifier(t1_feature,lambda_)
            #t1_domain_preds = t1_domain_output.argmax(1).cpu()
            #t1_domain_acc = np.mean((t1_domain_preds == 3).numpy())
            #t1_d_loss = moe_criterion(t1_domain_output,domain_labels[3])

            #D_loss = D_loss_src + t1_d_loss
            loss = C_loss
            D_loss = 0
            #epoch_D_loss += D_loss.item()
            epoch_C_loss += C_loss.item()
            #sum_trg_acc += t1_domain_acc
            #D_src_acc = (s1_domain_acc + s2_domain_acc + s3_domain_acc)/3.

            loss.backward()
            optimizer_encoder.step()
            optimizer_classifier.step()
            optimizer_domain_classifier.step()
            if (index+1) % 10 == 0:
                print('Iter [%d/%d] loss %.4f , D_loss %.4f ,Acc %.4f  ,Test Acc: %.4f'
                %(index+1, len(train_loader[0]), loss.item(),D_loss,s1_acc,test_acc))

        test_acc = 0.
        test_loss = 0.
        encoder.eval()
        domain_classifier.eval()
        classifier.eval()

        for index, (imgs,labels) in enumerate(test_loader):
            output_list = []
            loss_mtl = []
            imgs = Variable(imgs).to(device)
            labels = Variable(labels).to(device)
            hidden = encoder(imgs)
            output = classifier(hidden)
            preds = output.argmax(1).cpu()
            s1_acc = np.mean((preds == labels.cpu()).numpy())

            """
            for sthi in classifiers:
                output = sthi(hidden)
                output_list.append(output.cpu())
                loss = mtl_criterion(output, labels)
                loss_mtl.append(loss)


            output = torch.FloatTensor(np.array(output_list).sum(0))
            preds = output.argmax(1).cpu()
            s1_preds = output_list[0].argmax(1).cpu()
            s2_preds = output_list[1].argmax(1).cpu()
            s3_preds = output_list[2].argmax(1).cpu()
            acc = np.mean((preds == labels.cpu()).numpy())
            s1_acc = np.mean((s1_preds == labels.cpu()).numpy())
            s2_acc = np.mean((s2_preds == labels.cpu()).numpy())
            s3_acc = np.mean((s3_preds == labels.cpu()).numpy())
            if index == 0:
                print(acc)
            loss_mtl = sum(loss_mtl)
            loss = loss_mtl
            test_acc += acc
            test_loss += loss.item()
            """
        #print('Testing: loss %.4f,Acc %.4f ,s1 %.4f,s2 %.4f,s3 %.4f' %(test_loss/len(test_loader),test_acc/len(test_loader),s1_acc,s2_acc,s3_acc))

    return 0


if __name__ == '__main__':
    main()
