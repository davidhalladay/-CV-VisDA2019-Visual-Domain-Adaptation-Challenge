import random
import sys
import argparse
import numpy as np

import torch
from torch.autograd import Function
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F

class ReverseLayerFunction(Function):

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None


class Encoder(nn.Module):

    def __init__(self, input_size = 224, output_size = 1024):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            # state = (3,64,64)
            nn.Conv2d(3, 16, kernel_size = 3,padding = 1),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size = 3,padding = 1),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            # state = (32,32,32)
            nn.Conv2d(16, 32, kernel_size = 3,padding = 1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 32, kernel_size = 3,padding = 1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            # state = (64,16,16)
            nn.Conv2d(32, 64, kernel_size = 3,padding = 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            # state = (128,8,8)
            nn.Conv2d(64, 256, kernel_size = 3,padding = 1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            # state = (256,4,4)

        )

    def forward(self, X):
        x = self.model(X)
        output = x.view(-1, 256*4*4)
        return output

class Classifier(nn.Module):

    def __init__(self,feature_size):
        super(Classifier, self).__init__()
        self.feature_size = feature_size
        self.fc1 = nn.Linear(feature_size, 3072)
        self.bn1 = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 345)

    def forward(self, input):
        input=input.view(-1, self.feature_size)
        logits = F.relu(self.bn1(self.fc1(input)))
        logits = F.dropout(logits,training = self.training)
        logits = F.relu(self.bn2(self.fc2(logits)))
        logits = self.fc3(logits)

        return F.log_softmax(logits, 1)

class Moe_Classifier(nn.Module):

    def __init__(self,input = 345):
        super(Moe_Classifier, self).__init__()
        self.input = input
        self.fc1 = nn.Linear(int(input*3), 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 345)

    def forward(self, X,Y,Z):
        X=X.view(-1, self.input)
        Y=Y.view(-1, self.input)
        Z=Z.view(-1, self.input)
        input = torch.cat((X,Y,Z),1)
        logits = F.relu(self.bn1(self.fc1(input)))
        logits = F.dropout(logits,training = self.training)
        logits = F.relu(self.bn2(self.fc2(logits)))
        logits = self.fc3(logits)

        return F.log_softmax(logits, 1)

class Domain_classifier_0(nn.Module):

    def __init__(self,input_size,domain_size=4):
        super(Domain_classifier_0, self).__init__()
        self.input_size = input_size
        self.model = nn.Sequential(
            # state = (*,48 * 4 * 4)
            nn.Linear(input_size,3072),
            # state = (*,100)
            nn.BatchNorm1d(3072),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(3072, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(512, 2)
            # state = (*,2)
            # nn.LogSoftmax(dim = 1)
        )

    def forward(self, input, lambda_):
        input = input.view(-1,self.input_size)
        input = ReverseLayerFunction.apply(input, lambda_)
        output = self.model(input)
        output = torch.sigmoid(output)

        return output

class Domain_classifier_1(nn.Module):

    def __init__(self,input_size,domain_size=4):
        super(Domain_classifier_1, self).__init__()
        self.input_size = input_size
        self.model = nn.Sequential(
            # state = (*,48 * 4 * 4)
            nn.Linear(input_size,3072),
            # state = (*,100)
            nn.BatchNorm1d(3072),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(3072, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(512, 2)
            # state = (*,2)
            # nn.LogSoftmax(dim = 1)
        )

    def forward(self, input, lambda_):
        input = input.view(-1,self.input_size)
        input = ReverseLayerFunction.apply(input, lambda_)
        output = self.model(input)
        output = torch.sigmoid(output)

        return output

class Domain_classifier_2(nn.Module):

    def __init__(self,input_size,domain_size=4):
        super(Domain_classifier_2, self).__init__()
        self.input_size = input_size
        self.model = nn.Sequential(
            # state = (*,48 * 4 * 4)
            nn.Linear(input_size,3072),
            # state = (*,100)
            nn.BatchNorm1d(3072),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(3072, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(512, 2)
            # state = (*,2)
            # nn.LogSoftmax(dim = 1)
        )

    def forward(self, input, lambda_):
        input = input.view(-1,self.input_size)
        input = ReverseLayerFunction.apply(input, lambda_)
        output = self.model(input)
        output = torch.sigmoid(output)

        return output
