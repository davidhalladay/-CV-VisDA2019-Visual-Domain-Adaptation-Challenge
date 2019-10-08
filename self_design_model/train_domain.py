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

def main():
    # parameters
    learning_rate = 0.001
    num_epochs = 100
    batch_size = 50
    batch_size_test = 50
    feature_size = 2048 * 1 * 1


    # create the save log file
    print("Create the directory")
    if not os.path.exists("./save"):
        os.makedirs("./save")
    if not os.path.exists("./logfile"):
        os.makedirs("./logfile")
    if not os.path.exists("./logfile/MTL"):
        os.makedirs("./logfile/MTL")

    # load my Dataset
    inf_csv_path = ["./dataset_public/infograph/infograph_train.csv","./dataset_public/infograph/infograph_test.csv"]
    qdr_csv_path = ["./dataset_public/quickdraw/quickdraw_train.csv","./dataset_public/quickdraw/quickdraw_test.csv"]
    skt_csv_path = ["./dataset_public/sketch/sketch_train.csv","./dataset_public/sketch/sketch_test.csv"]
    rel_csv_path = ["./dataset_public/real/real_train.csv"]
    test_path = "./dataset_public/test"

    inf_train_dataset = Dataset.Dataset(csv_path = inf_csv_path[0],argu = True)
    inf_train_loader = DataLoader(inf_train_dataset ,batch_size = batch_size ,shuffle=True ,num_workers=1)
    qdr_train_dataset = Dataset.Dataset(csv_path = qdr_csv_path[0],argu = True)
    qdr_train_loader = DataLoader(qdr_train_dataset ,batch_size = batch_size ,shuffle=True ,num_workers=1)
    skt_train_dataset = Dataset.Dataset(csv_path = skt_csv_path[0],argu = True)
    skt_train_loader = DataLoader(skt_train_dataset ,batch_size = batch_size ,shuffle=True ,num_workers=1)
    rel_train_dataset = Dataset.Dataset(csv_path = rel_csv_path[0],sample = True,argu = True)
    rel_train_loader = DataLoader(rel_train_dataset ,batch_size = batch_size ,shuffle=True ,num_workers=1)

    valid_dataset = Dataset.Valid_Dataset(csv_path = rel_csv_path[0],sample = True)
    valid_loader = DataLoader(valid_dataset ,batch_size = batch_size ,shuffle=True ,num_workers=1)

    test_dataset = Dataset.Dataset(csv_path = test_path,mode = "test",filename = True)
    test_loader = DataLoader(test_dataset ,batch_size = batch_size_test ,shuffle=False ,num_workers=1)

    print('the source dataset has %d size.' % (len(rel_train_dataset)))
    print('the valid dataset has %d size.' % (len(valid_dataset)))
    print('the target dataset has %d size.' % (len(test_dataset)))
    print('the batch_size is %d' % (batch_size))

    # Pre-train models
    modules = list(models.resnet50(pretrained = True).children())[:-1]
    encoder = nn.Sequential(*modules)
    #encoder = model.Encoder()
    classifier = model.Classifier(feature_size)
    domain_classifier = model.Domain_classifier_0(feature_size,number_of_domain)

    # GPU enable
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        encoder = encoder.to(device)
        classifier = classifier.to(device)
        domain_classifier = domain_classifier.to(device)


    # setup optimizer
    """

    optimizer_encoder = optim.SGD(
            list(encoder.parameters()) +
            list(classifier.parameters()),lr=learning_rate,momentum=0.9)

    optimizer_classifier = optim.SGD(list(classifier.parameters()),lr=learning_rate,momentum=0.9)

    optimizer = optim.SGD([{'params': domain_classifier.parameters()}], lr= learning_rate,momentum=0.9)

    """
    optimizer_encoder = optim.Adam([{'params': encoder.parameters()},
                                  {'params': classifier.parameters()}], weight_decay = 1e-4 , lr= learning_rate)
    optimizer_classifier = optim.Adam(classifier.parameters(), weight_decay = 1e-4 , lr= learning_rate)
    optimizer = optim.Adam([{'params': domain_classifier.parameters()}], lr= learning_rate, weight_decay = 1e-4 )


    #Lossfunction
    moe_criterion = nn.CrossEntropyLoss()
    mtl_criterion = nn.NLLLoss()

    D_loss_list = []
    L_loss_list = []
    sum_src_acc_list = []
    sum_trg_acc_list = []
    sum_label_acc_loist = []
    sum_test_acc_list = []
    print("Starting training...")
    best_acc = 0.
    valid_acc = 0.0
    for epoch in range(num_epochs):
        print("Epoch:", epoch+1)

        encoder.train()
        classifier.train()
        domain_classifier.train()

        epoch_D_loss = 0.0
        epoch_C_loss = 0.0
        sum_src_acc = 0.0
        sum_trg_acc = 0.0
        sum_label_acc = 0.0
        sum_test_acc = 0.0
        tmp_valid_acc = 0.0

        if (epoch+1) in [5,10,15,20,25,30,35,40,45]:
            for optimizer_t in optimizer.param_groups:
                optimizer_t['lr'] /= 1.7

        train_loader = [inf_train_loader,qdr_train_loader,skt_train_loader,rel_train_loader]
        len_loader = min([len(train_loader[0]),len(train_loader[1]),len(train_loader[2]),len(train_loader[3])])

        for index, (inf,qdr,skt,rel,test) in enumerate(zip(train_loader[0],train_loader[1],train_loader[2],train_loader[3],valid_loader)):

            optimizer_encoder.zero_grad()
            optimizer_classifier.zero_grad()
            optimizer.zero_grad()

            # colculate the lambda_
            p = (index + len_loader * epoch)/(len_loader * num_epochs)
            lambda_ = 2.0 / (1. + np.exp(-10 * p)) - 1.0

            s1_imgs , s1_labels = skt
            s2_imgs , s2_labels = inf
            s3_imgs , s3_labels = inf
            test_imgs , test_labels = test
            t1_imgs , _ = rel
            from_s2_labels = Variable(torch.LongTensor([0 for i in range(len(s3_imgs))])).to(device)

            from_t1_labels = Variable(torch.LongTensor([1 for i in range(len(t1_imgs))])).to(device)


            s1_imgs = Variable(s1_imgs).to(device) ; s1_labels = Variable(s1_labels.view(-1)).to(device)
            s2_imgs = Variable(s2_imgs).to(device) ; s2_labels = Variable(s2_labels.view(-1)).to(device)
            s3_imgs = Variable(s3_imgs).to(device) ; s3_labels = Variable(s3_labels.view(-1)).to(device)
            test_imgs = Variable(test_imgs).to(device) ; test_labels = Variable(test_labels.view(-1)).to(device)
            t1_imgs = Variable(t1_imgs).to(device)

            s2_feature = encoder(s2_imgs)
            test_feature = encoder(test_imgs)
            t1_feature = encoder(t1_imgs)

            test_output = classifier(test_feature)
            test_preds = test_output.argmax(1).cpu()
            test_acc = np.mean((test_preds.detach().cpu() == test_labels.cpu()).numpy())

            s2_output = classifier(s2_feature)
            s2_preds = s2_output.argmax(1).cpu()
            s2_acc = np.mean((s2_preds.detach().cpu() == s2_labels.cpu()).numpy())
            s2_c_loss = mtl_criterion(s2_output,s2_labels)
            mtl_loss = s2_c_loss

            # Domain_classifier network with source domain (loss_adv)
            s2_domain_output = domain_classifier(s2_feature,lambda_)

            s2_domain_acc = np.mean((s2_domain_output.argmax(1).cpu() <= 0.5).numpy())
            s2_d_loss = moe_criterion(s2_domain_output,from_s2_labels)
            D_loss_src = s2_d_loss
            #print(D_loss_src.item())

            # Domain_classifier network with target domain (loss_adv)
            t1_domain_0_output = domain_classifier(t1_feature,lambda_)

            t1_domain_0_acc = np.mean((t1_domain_0_output.argmax(1).cpu() > 0.5).numpy())
            D0_loss_trg = moe_criterion(t1_domain_0_output,from_t1_labels)
            D_loss_trg = D0_loss_trg
            if (index+1) % 100 == 0:
                print(s2_domain_output.argmax(1).cpu())
                print(t1_domain_0_output.argmax(1).cpu())
            adv_loss = D_loss_src + D_loss_trg

            loss = 1 * mtl_loss + 1 * adv_loss

            D_trg_acc = t1_domain_0_acc
            D_src_acc = s2_domain_acc

            #mtl_loss.backward()
            #adv_loss.backward()
            loss.backward()

            optimizer.step()
            optimizer_encoder.step()
            optimizer_classifier.step()

            if (index+1) % 10 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] C_loss : %.4f D_loss %.4f arc: %.4f trg: %.4f ,LR: %.5f'
                %(epoch+1, num_epochs,index+1, len_loader,mtl_loss.item(), adv_loss.item(),D_loss_src.item(),D_loss_trg.item(),optimizer.param_groups[0]['lr']))
                print("====> Domain Acc: %.4f %.4f Test: %.4f"%(D_src_acc,D_trg_acc,test_acc))



        print('Validing: Acc %.4f ' %(tmp_valid_acc/len_loader))
        if tmp_valid_acc/len_loader > best_acc :
            best_acc = tmp_valid_acc/len_loader
            print('Find best: Acc %.6f ' %(best_acc))
        print('Best: Acc %.6f ' %(best_acc))


    return 0


if __name__ == '__main__':
    main()
