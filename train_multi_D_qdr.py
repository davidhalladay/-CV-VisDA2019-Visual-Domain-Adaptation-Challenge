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

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path,map_location = "cuda")
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)

def main():
    # parameters
    learning_rate = 0.001
    num_epochs = 30
    batch_size = 50
    batch_size_test = 20
    feature_size = 2048 * 1 * 1
    #learning_rate = float(sys.argv[1])

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
    inf_train_loader = DataLoader(inf_train_dataset ,batch_size = batch_size ,shuffle=True ,num_workers=4)
    qdr_train_dataset = Dataset.Dataset(csv_path = qdr_csv_path[0],argu = True)
    qdr_train_loader = DataLoader(qdr_train_dataset ,batch_size = batch_size ,shuffle=True ,num_workers=4)
    skt_train_dataset = Dataset.Dataset(csv_path = skt_csv_path[0],argu = True)
    skt_train_loader = DataLoader(skt_train_dataset ,batch_size = batch_size ,shuffle=True ,num_workers=4)
    rel_train_dataset = Dataset.Dataset(csv_path = rel_csv_path[0],sample = True,argu = True)
    rel_train_loader = DataLoader(rel_train_dataset ,batch_size = batch_size ,shuffle=True ,num_workers=4)

    valid_dataset = Dataset.Valid_Dataset(csv_path = qdr_csv_path[1])
    valid_loader = DataLoader(valid_dataset ,batch_size = batch_size_test ,shuffle=True ,num_workers=4)
    test_loader = valid_loader
    #test_dataset = Dataset.Dataset(csv_path = test_path,mode = "test",filename = True)
    #test_loader = DataLoader(test_dataset ,batch_size = batch_size_test ,shuffle=False ,num_workers=4)

    print('the source dataset has %d size.' % (len(rel_train_dataset)))
    print('the valid dataset has %d size.' % (len(valid_dataset)))
    print('the target dataset has %d size.' % (len(test_loader)))
    print('the batch_size is %d' % (batch_size))

    # Pre-train models
    modules = list(models.resnet50(pretrained = True).children())[:-1]
    encoder = nn.Sequential(*modules)
    #encoder = model.Encoder()
    classifier = model.Classifier(feature_size)
    domain_classifier_0 = model.Domain_classifier_0(feature_size,number_of_domain)
    domain_classifier_1 = model.Domain_classifier_1(feature_size,number_of_domain)
    domain_classifier_2 = model.Domain_classifier_2(feature_size,number_of_domain)

    # GPU enable
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        encoder = encoder.to(device)
        classifier = classifier.to(device)
        domain_classifier_0 = domain_classifier_0.to(device)
        domain_classifier_1 = domain_classifier_1.to(device)
        domain_classifier_2 = domain_classifier_2.to(device)


    # setup optimizer

    optimizer_encoder = optim.SGD(list(encoder.parameters())+
                                list(classifier.parameters()),lr=learning_rate,momentum=0.9)

    optimizer_classifier = optim.SGD(list(classifier.parameters()),lr=learning_rate,momentum=0.9)

    optimizer_domain = optim.Adam([
                                  {'params': domain_classifier_0.parameters()},
                                  {'params': domain_classifier_1.parameters()},
                                  {'params': domain_classifier_2.parameters()}], lr= learning_rate, weight_decay = 1e-4)
    """
    optimizer_encoder = optim.Adam([{'params': encoder.parameters()},
                                  {'params': classifier.parameters()}], weight_decay = 1e-4 , lr= learning_rate)
    optimizer_classifier = optim.Adam(classifier.parameters(), weight_decay = 1e-4 , lr= learning_rate)
    optimizer_domain = optim.Adam([{'params': domain_classifier_0.parameters()},
                                  {'params': domain_classifier_1.parameters()},
                                  {'params': domain_classifier_2.parameters()}], lr= learning_rate, weight_decay = 1e-4 )
    """



    #loading models
    #if input("Loading pre- file?(T/F)") == "T":
    #    encoder_path = sys.argv[2]
    #    classifier_path = sys.argv[3]
    #    load_checkpoint(encoder_path,encoder)
    #    load_checkpoint(classifier_path,classifier)

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
        domain_classifier_0.train()
        domain_classifier_1.train()
        domain_classifier_2.train()

        epoch_D_loss = 0.0
        epoch_C_loss = 0.0
        domain_acc_src = 0.0
        domain_acc_trg = 0.0

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
        sum_acc_1 = 0.;sum_acc_2 = 0.;sum_acc_3 = 0.
        for index, (inf,qdr,skt,rel,test) in enumerate(zip(train_loader[0],train_loader[1],train_loader[2],train_loader[3],valid_loader)):

            optimizer_classifier.zero_grad()
            optimizer_encoder.zero_grad()
            optimizer_domain.zero_grad()

            # colculate the lambda_
            p = (index + len_loader * epoch)/(len_loader * num_epochs)
            lambda_ = 2.0 / (1. + np.exp(-10 * p)) - 1.0

            s1_imgs , s1_labels = skt
            s2_imgs , s2_labels = inf
            s3_imgs , s3_labels = rel
            t1_imgs , _ = qdr

            from_s1_labels = Variable(torch.LongTensor([0 for i in range(len(s1_imgs))])).to(device)
            from_s2_labels = Variable(torch.LongTensor([0 for i in range(len(s2_imgs))])).to(device)
            from_s3_labels = Variable(torch.LongTensor([0 for i in range(len(s3_imgs))])).to(device)
            from_t1_labels = Variable(torch.LongTensor([1 for i in range(len(t1_imgs))])).to(device)


            s1_imgs = Variable(s1_imgs).to(device) ; s1_labels = Variable(s1_labels.view(-1)).to(device)
            s2_imgs = Variable(s2_imgs).to(device) ; s2_labels = Variable(s2_labels.view(-1)).to(device)
            s3_imgs = Variable(s3_imgs).to(device) ; s3_labels = Variable(s3_labels.view(-1)).to(device)
            t1_imgs = Variable(t1_imgs).to(device)

            s1_feature = encoder(s1_imgs)
            s2_feature = encoder(s2_imgs)
            s3_feature = encoder(s3_imgs)
            t1_feature = encoder(t1_imgs)

            if len(s1_feature) < 2 or len(s2_feature) < 2 or len(s3_feature) < 2 or len(t1_feature) < 2:
                break

            # Training Classifier network (loss_mtl)
            s1_output = classifier(s1_feature)
            s2_output = classifier(s2_feature)
            s3_output = classifier(s3_feature)

            s1_preds = s1_output.argmax(1).cpu()
            s2_preds = s2_output.argmax(1).cpu()
            s3_preds = s3_output.argmax(1).cpu()
            s1_acc = np.mean((s1_preds.detach().cpu() == s1_labels.cpu()).numpy())
            s2_acc = np.mean((s2_preds.detach().cpu() == s2_labels.cpu()).numpy())
            s3_acc = np.mean((s3_preds.detach().cpu() == s3_labels.cpu()).numpy())
            s1_c_loss = mtl_criterion(s1_output,s1_labels)
            s2_c_loss = mtl_criterion(s2_output,s2_labels)
            s3_c_loss = mtl_criterion(s3_output,s3_labels)
            mtl_loss = 1*(1* s1_c_loss + 1. * s2_c_loss + 1. *s3_c_loss)
            s_acc = s1_acc + s2_acc + s3_acc

            # Domain_classifier network with source domain (loss_adv)
            s1_domain_output = domain_classifier_0(s1_feature,lambda_)
            s2_domain_output = domain_classifier_1(s2_feature,lambda_)
            s3_domain_output = domain_classifier_2(s3_feature,lambda_)
            #if index == 10:
            #    print(s1_domain_preds)
            s1_domain_acc = np.mean((s1_domain_output.argmax(1).cpu() == from_s1_labels.cpu()).numpy())
            s2_domain_acc = np.mean((s2_domain_output.argmax(1).cpu() == from_s2_labels.cpu()).numpy())
            s3_domain_acc = np.mean((s3_domain_output.argmax(1).cpu() == from_s3_labels.cpu()).numpy())
            #print(s1_domain_output.shape)
            #print(s1_domain_output[0])
            s1_d_loss = moe_criterion(s1_domain_output,from_s1_labels)
            s2_d_loss = moe_criterion(s2_domain_output,from_s2_labels)
            s3_d_loss = moe_criterion(s3_domain_output,from_s3_labels)
            #D_loss_src = 1 * s1_d_loss + s2_d_loss + 1* s3_d_loss
            #print(D_loss_src.item())

            # Domain_classifier network with target domain (loss_adv)
            t1_domain_0_output = domain_classifier_0(t1_feature,lambda_)
            t1_domain_1_output = domain_classifier_1(t1_feature,lambda_)
            t1_domain_2_output = domain_classifier_2(t1_feature,lambda_)

            t1_domain_0_acc = np.mean((t1_domain_0_output.argmax(1).cpu() == from_t1_labels.cpu()).numpy())
            t1_domain_1_acc = np.mean((t1_domain_1_output.argmax(1).cpu() == from_t1_labels.cpu()).numpy())
            t1_domain_2_acc = np.mean((t1_domain_2_output.argmax(1).cpu() == from_t1_labels.cpu()).numpy())
            D0_loss_trg = moe_criterion(t1_domain_0_output,from_t1_labels)
            D1_loss_trg = moe_criterion(t1_domain_1_output,from_t1_labels)
            D2_loss_trg = moe_criterion(t1_domain_2_output,from_t1_labels)
            D_loss_trg = (1 * D0_loss_trg + D1_loss_trg + 1 * D2_loss_trg)/2.9


            D_s1t1_loss = s1_d_loss
            D_s2t1_loss = s2_d_loss
            D_s3t1_loss = s3_d_loss
            if D_s1t1_loss > D_s2t1_loss and D_s1t1_loss > D_s3t1_loss:
                adv_loss = D_s1t1_loss + D_loss_trg
            if D_s2t1_loss > D_s1t1_loss and D_s2t1_loss > D_s3t1_loss:
                adv_loss = D_s2t1_loss + D_loss_trg
            if D_s3t1_loss > D_s1t1_loss and D_s3t1_loss > D_s2t1_loss:
                adv_loss = D_s3t1_loss + D_loss_trg



            #adv_loss = D_loss_src + D_loss_trg

            loss = 1. * mtl_loss + adv_loss

            epoch_D_loss += adv_loss.item()
            epoch_C_loss += mtl_loss.item()
            D_trg_acc = (t1_domain_0_acc + t1_domain_1_acc + t1_domain_2_acc)/3.
            D_src_acc = (s1_domain_acc + s2_domain_acc + s3_domain_acc)/3.
            sum_acc_1 += s1_acc
            sum_acc_2 += s2_acc
            sum_acc_3 += s3_acc
            domain_acc_src += D_src_acc
            domain_acc_trg += D_trg_acc
            #mtl_loss.backward()
            #adv_loss.backward()
            loss.backward()
            optimizer_classifier.step()
            optimizer_encoder.step()
            optimizer_domain.step()

            if (index+1) % 10 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] C_loss %.4f , D_loss %.4f ,LR: %.5f'
                %(epoch+1, num_epochs,index+1, len_loader, epoch_C_loss/(index+1),epoch_D_loss/(index+1),optimizer_domain.param_groups[0]['lr']))
                print("====> Acc %.4f %.4f %.4f Domain Acc: %.4f %.4f ,Test Acc: %.4f"%(sum_acc_1/(index+1),sum_acc_2/(index+1),sum_acc_3/(index+1),domain_acc_src/(index+1),domain_acc_trg/(index+1),tmp_valid_acc/(index+1)))


            # Testing
            encoder.eval()
            classifier.eval()
            with torch.no_grad():
                test_imgs , test_labels = test
                test_imgs = Variable(test_imgs).to(device) ; test_labels = Variable(test_labels.view(-1)).to(device)
                test_feature = encoder(test_imgs)

                test_output = classifier(test_feature)
                test_preds = test_output.argmax(1).cpu()
                test_acc = np.mean((test_preds.detach().cpu() == test_labels.cpu()).numpy())
                tmp_valid_acc += test_acc

            encoder.train()
            classifier.train()

        s1_avg_acc = sum_acc_1/len_loader
        s2_avg_acc = sum_acc_2/len_loader
        s3_avg_acc = sum_acc_3/len_loader


        print('Validing: Acc %.4f ' %(tmp_valid_acc/len_loader))
        print('Avg sur: Acc %.4f,%.4f,%.4f ' %(s1_avg_acc,s2_avg_acc,s3_avg_acc))
        if tmp_valid_acc/len_loader > best_acc :
            best_acc = tmp_valid_acc/len_loader
            print('Find best: Acc %.6f ' %(best_acc))

            save_checkpoint('./save/encoder-%.4f-%.4f-%.4f.pth' % (s1_avg_acc,s2_avg_acc,s3_avg_acc) , encoder)
            save_checkpoint('./save/domain-%.4f-%.4f.pth' % (domain_acc_src/(index+1),domain_acc_trg/(index+1)) , domain_classifier_0)
            save_checkpoint('./save/classifier-%.4f-%.4f-%.4f.pth' % (s1_avg_acc,s2_avg_acc,s3_avg_acc) , classifier)
        print('Best: Acc %.6f Avg sur: Acc %.4f,%.4f,%.4f' %(best_acc,s1_avg_acc,s2_avg_acc,s3_avg_acc))

        if tmp_valid_acc/len_loader > 0.3:
            best_acc = tmp_valid_acc/len_loader
            print('Find best: Acc %.6f ' %(best_acc))

            save_checkpoint('./save/encoder--%.4f.pth' % (best_acc) , encoder)
            save_checkpoint('./save/classifier--%.4f.pth' % (best_acc) , classifier)

            test_acc = 0.
            test_loss = 0.
            encoder.eval()
            classifier.eval()
            domain_classifier_0.eval()
            domain_classifier_1.eval()
            domain_classifier_2.eval()
            filename_save = []
            output_save = []
            with torch.no_grad():
                for index, (imgs,filenames) in enumerate(test_loader):
                    output_list = []
                    loss_mtl = []
                    imgs = Variable(imgs).to(device)
                    #labels = Variable(labels.view(-1)).to(device)
                    hidden = encoder(imgs)
                    output = classifier(hidden)
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
