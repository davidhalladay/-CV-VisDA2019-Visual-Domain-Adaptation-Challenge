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
    batch_size = 20
    batch_size_test = 20
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
    modules = list(models.resnet152(pretrained = True).children())[:-1]
    encoder = nn.Sequential(*modules)
    #encoder = model.Encoder()
    classifier_0 = model.Classifier(feature_size)
    classifier_1 = model.Classifier(feature_size)
    classifier_2 = model.Classifier(feature_size)
    moe_classifier = model.Moe_Classifier()
    domain_classifier_0 = model.Domain_classifier(feature_size,number_of_domain)
    domain_classifier_1 = model.Domain_classifier(feature_size,number_of_domain)
    domain_classifier_2 = model.Domain_classifier(feature_size,number_of_domain)

    # GPU enable
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        encoder = encoder.to(device)
        classifier_0 = classifier_0.to(device)
        classifier_1 = classifier_1.to(device)
        classifier_2 = classifier_2.to(device)
        moe_classifier = moe_classifier.to(device)
        domain_classifier_0 = domain_classifier_0.to(device)
        domain_classifier_1 = domain_classifier_1.to(device)
        domain_classifier_2 = domain_classifier_2.to(device)


    # setup optimizer
    """
    optimizer_encoder = optim.SGD(
            list(encoder.parameters()),lr=learning_rate,momentum=0.9)

    optimizer_classifier = optim.SGD(
            list(classifier_0.parameters())+
            list(classifier_1.parameters())+
            list(classifier_2.parameters())+
            list(moe_classifier.parameters()),lr=learning_rate,momentum=0.9)

    optimizer_domain = optim.SGD([{'params': domain_classifier_0.parameters()},
                                  {'params': domain_classifier_1.parameters()},
                                  {'params': domain_classifier_2.parameters()}], lr= learning_rate,momentum=0.9)
    """
    optimizer_encoder = optim.Adam([{'params': encoder.parameters()},
                                  ], weight_decay = 1e-4 , lr= learning_rate)
    optimizer_classifier = optim.Adam(list(classifier_0.parameters())+
                                      list(classifier_1.parameters())+
                                      list(classifier_2.parameters())+
                                      list(moe_classifier.parameters()), weight_decay = 1e-4 , lr= learning_rate)
    optimizer_domain = optim.Adam([{'params': domain_classifier_0.parameters()},
                                  {'params': domain_classifier_1.parameters()},
                                  {'params': domain_classifier_2.parameters()}], lr= learning_rate, weight_decay = 1e-4 )


    #Lossfunction
    moe_criterion = nn.BCELoss()
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
        classifier_0.train()
        classifier_1.train()
        classifier_2.train()
        moe_classifier.train()
        domain_classifier_0.train()
        domain_classifier_1.train()
        domain_classifier_2.train()

        epoch_D_loss = 0.0
        epoch_C_loss = 0.0
        epoch_C_moe_loss = 0.0
        sum_src_acc = 0.0
        sum_trg_acc = 0.0
        sum_label_acc = 0.0
        sum_test_acc = 0.0
        tmp_valid_acc = 0.0

        if (epoch+1) in [5,10,15,20,25,30,35,40,45]:
            for optimizer_t in optimizer_domain.param_groups:
                optimizer_t['lr'] /= 1.7
            optimizer_encoder.param_groups[0]['lr'] /= 1.7
            optimizer_classifier.param_groups[0]['lr'] /= 1.7

        train_loader = [inf_train_loader,qdr_train_loader,skt_train_loader,rel_train_loader]
        len_loader = min([len(train_loader[0]),len(train_loader[1]),len(train_loader[2]),len(train_loader[3])])

        for index, (inf,qdr,skt,rel,test) in enumerate(zip(train_loader[0],train_loader[1],train_loader[2],train_loader[3],valid_loader)):

            optimizer_classifier.zero_grad()
            optimizer_encoder.zero_grad()
            optimizer_domain.zero_grad()

            # colculate the lambda_
            p = (index + len_loader * epoch)/(len_loader * num_epochs)
            lambda_ = 5.0 / (1. + np.exp(-10 * p)) - 1.0

            s1_imgs , s1_labels = skt
            s2_imgs , s2_labels = qdr
            s3_imgs , s3_labels = inf
            t1_imgs , _ = rel
            from_s1_labels = Variable(torch.zeros(len(s1_imgs))).to(device)
            from_s2_labels = Variable(torch.zeros(len(s2_imgs))).to(device)
            from_s3_labels = Variable(torch.zeros(len(s3_imgs))).to(device)
            from_t1_labels = Variable(torch.ones(len(t1_imgs))).to(device)


            s1_imgs = Variable(s1_imgs).to(device) ; s1_labels = Variable(s1_labels.view(-1)).to(device)
            s2_imgs = Variable(s2_imgs).to(device) ; s2_labels = Variable(s2_labels.view(-1)).to(device)
            s3_imgs = Variable(s3_imgs).to(device) ; s3_labels = Variable(s3_labels.view(-1)).to(device)
            t1_imgs = Variable(t1_imgs).to(device)

            s1_feature = encoder(s1_imgs)
            s2_feature = encoder(s2_imgs)
            s3_feature = encoder(s3_imgs)
            t1_feature = encoder(t1_imgs)

            # Testing
            test_imgs , test_labels = test
            test_imgs = Variable(test_imgs).to(device) ; test_labels = Variable(test_labels.view(-1)).to(device)
            test_feature = encoder(test_imgs)
            test_output_0 = classifier_0(test_feature)
            test_output_1 = classifier_1(test_feature)
            test_output_2 = classifier_2(test_feature)
            test_output = moe_classifier(test_output_0,test_output_1,test_output_2)
            test_preds = test_output.argmax(1).cpu()
            test_acc = np.mean((test_preds.detach().cpu() == test_labels.cpu()).numpy())
            tmp_valid_acc += test_acc

            # Training Classifier network (loss_mtl)
            s1_output = classifier_0(s1_feature)
            s2_output = classifier_1(s2_feature)
            s3_output = classifier_2(s3_feature)

            s1_preds = s1_output.argmax(1).cpu()
            s2_preds = s2_output.argmax(1).cpu()
            s3_preds = s3_output.argmax(1).cpu()
            s1_acc = np.mean((s1_preds.detach().cpu() == s1_labels.cpu()).numpy())
            s2_acc = np.mean((s2_preds.detach().cpu() == s2_labels.cpu()).numpy())
            s3_acc = np.mean((s3_preds.detach().cpu() == s3_labels.cpu()).numpy())
            s1_c_loss = mtl_criterion(s1_output,s1_labels)
            s2_c_loss = mtl_criterion(s2_output,s2_labels)
            s3_c_loss = mtl_criterion(s3_output,s3_labels)
            mtl_loss = s1_c_loss + s2_c_loss + s3_c_loss

            # Domain_classifier network with source domain (loss_adv)
            s1_domain_output = domain_classifier_0(s1_feature,lambda_)
            s2_domain_output = domain_classifier_1(s2_feature,lambda_)
            s3_domain_output = domain_classifier_2(s3_feature,lambda_)

            s1_domain_acc = np.mean((s1_domain_output.detach().cpu() <= 0.5).numpy())
            s2_domain_acc = np.mean((s2_domain_output.detach().cpu() <= 0.5).numpy())
            s3_domain_acc = np.mean((s3_domain_output.detach().cpu() <= 0.5).numpy())

            s1_d_loss = moe_criterion(s1_domain_output,from_s1_labels)
            s2_d_loss = moe_criterion(s2_domain_output,from_s2_labels)
            s3_d_loss = moe_criterion(s3_domain_output,from_s3_labels)
            D_loss_src = s1_d_loss + s2_d_loss + s3_d_loss
            #print(D_loss_src.item())

            # Domain_classifier network with target domain (loss_adv)
            t1_domain_0_output = domain_classifier_0(t1_feature,lambda_)
            t1_domain_1_output = domain_classifier_1(t1_feature,lambda_)
            t1_domain_2_output = domain_classifier_2(t1_feature,lambda_)

            t1_domain_0_acc = np.mean((t1_domain_0_output.detach().cpu() > 0.5).numpy())
            t1_domain_1_acc = np.mean((t1_domain_1_output.detach().cpu() > 0.5).numpy())
            t1_domain_2_acc = np.mean((t1_domain_2_output.detach().cpu() > 0.5).numpy())
            D0_loss_trg = moe_criterion(t1_domain_0_output,from_t1_labels)
            D1_loss_trg = moe_criterion(t1_domain_1_output,from_t1_labels)
            D2_loss_trg = moe_criterion(t1_domain_2_output,from_t1_labels)
            D_loss_trg = D0_loss_trg + D1_loss_trg + D2_loss_trg
            adv_loss = D_loss_src + D_loss_trg

            # Moe combination
            s1_output_0 = classifier_0(s1_feature)
            s1_output_1 = classifier_1(s1_feature)
            s1_output_2 = classifier_2(s1_feature)
            s2_output_0 = classifier_0(s2_feature)
            s2_output_1 = classifier_1(s2_feature)
            s2_output_2 = classifier_2(s2_feature)
            s3_output_0 = classifier_0(s3_feature)
            s3_output_1 = classifier_1(s3_feature)
            s3_output_2 = classifier_2(s3_feature)
            s1_output_moe = moe_classifier(s1_output_0,s1_output_1,s1_output_2)
            s2_output_moe = moe_classifier(s2_output_0,s2_output_1,s2_output_2)
            s3_output_moe = moe_classifier(s3_output_0,s3_output_1,s3_output_2)
            s1_preds_moe = s1_output_moe.argmax(1).cpu()
            s2_preds_moe = s2_output_moe.argmax(1).cpu()
            s3_preds_moe = s3_output_moe.argmax(1).cpu()
            s1_acc = np.mean((s1_preds_moe.detach().cpu() == s1_labels.cpu()).numpy())
            s2_acc = np.mean((s2_preds_moe.detach().cpu() == s2_labels.cpu()).numpy())
            s3_acc = np.mean((s3_preds_moe.detach().cpu() == s3_labels.cpu()).numpy())
            s1_c_loss_moe = mtl_criterion(s1_output_moe,s1_labels)
            s2_c_loss_moe = mtl_criterion(s2_output_moe,s2_labels)
            s3_c_loss_moe = mtl_criterion(s3_output_moe,s3_labels)
            moe_loss = s1_c_loss_moe + s2_c_loss_moe + s3_c_loss_moe
            moe_acc = (s1_acc + s2_acc + s3_acc)/3.

            loss = 0.4 * mtl_loss + 0.4 * moe_loss + 2 * adv_loss

            epoch_D_loss += adv_loss.item()
            epoch_C_loss += mtl_loss.item()
            epoch_C_moe_loss += moe_loss.item()
            D_trg_acc = (t1_domain_0_acc + t1_domain_1_acc + t1_domain_2_acc)/3.
            D_src_acc = (s1_domain_acc + s2_domain_acc + s3_domain_acc)/3.

            loss.backward()
            optimizer_domain.step()
            optimizer_classifier.step()
            optimizer_encoder.step()

            if (index+1) % 10 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] C_loss %.4f , D_loss %.4f ,LR: %.5f'
                %(epoch+1, num_epochs,index+1, len_loader, mtl_loss.item(),adv_loss.item(),optimizer_encoder.param_groups[0]['lr']))
                print("====> Acc %.4f %.4f %.4f Domain Acc: %.4f %.4f ,Test Acc: %.4f"%(s1_acc,s2_acc,s3_acc,D_src_acc,D_trg_acc,test_acc))
                print("====> Loss_moe %.4f ,Moe acc %.4f"%(moe_loss.item(),moe_acc))

        print('Validing: Acc %.4f ' %(tmp_valid_acc/len_loader))
        if tmp_valid_acc/len_loader > best_acc :
            best_acc = tmp_valid_acc/len_loader
            print('Find best: Acc %.6f ' %(best_acc))
        print('Best: Acc %.6f ' %(best_acc))

        if tmp_valid_acc/len_loader >= best_acc and epoch+1 > 5 and tmp_valid_acc/len_loader > 0.3:
            best_acc = tmp_valid_acc/len_loader
            print('Find best: Acc %.6f ' %(best_acc))

            save_checkpoint('./save/encoder-multiD-rel-%.4f-%03i.pth' % (best_acc,epoch) , encoder)
            save_checkpoint('./save/classifier-multiD-rel-%.4f-%03i.pth' % (best_acc,epoch) , moe_classifier)
            save_checkpoint('./save/classifier-multiD-rel-%.4f-%03i.pth' % (best_acc,epoch) , classifier_0)
            save_checkpoint('./save/classifier-multiD-rel-%.4f-%03i.pth' % (best_acc,epoch) , classifier_1)
            save_checkpoint('./save/classifier-multiD-rel-%.4f-%03i.pth' % (best_acc,epoch) , classifier_2)

            test_acc = 0.
            test_loss = 0.
            encoder.eval()
            moe_classifier.eval()
            classifier_0.eval()
            classifier_1.eval()
            classifier_2.eval()
            domain_classifier_0.eval()
            domain_classifier_1.eval()
            domain_classifier_2.eval()
            filename_save = []
            output_save = []
            for index, (imgs,filenames) in enumerate(test_loader):
                output_list = []
                loss_mtl = []
                imgs = Variable(imgs).to(device)
                #labels = Variable(labels.view(-1)).to(device)
                hidden = encoder(imgs)
                tmp_0 = classifier_0(hidden)
                tmp_1 = classifier_1(hidden)
                tmp_2 = classifier_2(hidden)
                output = moe_classifier(tmp_0,tmp_1,tmp_2)
                preds = output.argmax(1).cpu()
                #acc = np.mean((preds.detach().cpu() == labels.cpu()).numpy())
                for filename in filenames:
                    filename_save.append(filename[-10:])
                for out in preds.detach().cpu():
                    output_save.append(out)
                #test_acc += acc
                #if index % 500 == 0:
                    #print(acc)

            # save csv
            file = open("./submission/multi_01_rel.csv","w")
            file.write("image_name,label\n")
            for i,(filename,data) in enumerate(zip(filename_save,output_save)):
                file.write("test/%s,%d\n"%(filename,int(data)))

    return 0


if __name__ == '__main__':
    main()
