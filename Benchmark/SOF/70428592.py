
import torch
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
##from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer
#from torchtnt.framework.callback import Callback
from torchtnt.framework import train
class CNN(nn.Module):

# Contructor
    def __init__(self):
        super(CNN, self).__init__()
        # Conv1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.maxpool1=nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv2
        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5,stride=1, padding=0)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.maxpool2=nn.MaxPool2d(kernel_size=2, stride=2)  
        
        # Conv3
        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5,stride=1, padding=0)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.maxpool3=nn.MaxPool2d(kernel_size=2, stride=2)
    
    
        # FCL 1
        self.fc1 = nn.Linear(in_features=128 * 27 * 27, out_features=500)
        self.bn_fc1 = nn.BatchNorm1d(500)
        
        # FCL 2
        self.fc2 = nn.Linear(in_features=500, out_features=500)
        self.bn_fc2 = nn.BatchNorm1d(500)
        
        # FCL3
        self.fc3 = nn.Linear(in_features=500, out_features=1)
    

    

# Prediction
def forward(self, x):
    # conv1
    x = self.cnn1(x)
    x = self.conv1_bn(x)
    x = torch.relu(x)
    x = self.maxpool1(x)
    # conv2
    x = self.cnn2(x)
    x = self.conv2_bn(x)
    x = torch.relu(x)
    x = self.maxpool2(x)
    # conv3
    x = self.cnn3(x)
    x = self.conv3_bn(x)
    x = torch.relu(x)
    x = self.maxpool3(x)
    
    # Fcl1
    x = x.view(x.size(0), -1)
    x = self.fc1(x)
    x = self.bn_fc1(x)
    x = torch.relu(x)
    # Fcl2
    x = self.fc2(x)
    x = self.bn_fc2(x)
    x = torch.relu(x)
    # final fcl
    x = self.fc3(x)
    x = torch.sigmoid(x)
   

def train_model(model,train_loader,test_loader,optimizer,n_epochs=5):
    
#global variable 
        N_test=len(dataset_val)
        accuracy_list=[]

        loss_list=[]
        for epoch in range(n_epochs):
            cost = 0
            model.train()
            print(f"Epoch: {epoch + 1}")
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                z = model(x)
                y = y.unsqueeze(-1)
                y = y.float()
                loss = criterion(z, y)
                loss.backward()
                optimizer.step()
                cost+=loss.item()

            correct=0
            model.eval()
            #perform a prediction on the validation  data  
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(device), y_test.to(device)
                
                z = model(x_test)
                _, yhat = torch.max(z.data, 1)
                correct += (yhat == y_test).sum().item()

            accuracy = correct / N_test
            accuracy_list.append(accuracy)
            loss_list.append(cost)
            print(f"------>  loss: {round(cost, 8)}, accuracy_val: %{accuracy * 100}")

        
        return accuracy_list, loss_lis