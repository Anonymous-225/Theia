import numpy as np
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from torchvision import datasets,transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
#from keras.datasets import mnist
from torchvision import datasets



train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]), 
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
)

print(train_data)

class ConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d((2,2))
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d((2,2))
        #self.fc1 = nn.Linear(5*5*128, 1024) 
        self.fc1 = nn.Linear(128, 1024) 
        self.fc1relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 2048)
        self.fc2relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc3 = nn.Linear(2048, 10)
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        x= self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x =self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x =self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        print('Here',x.shape)
        #x = torch.flatten(x, 1)
        #x = torch.randn(1,28,28).view(-1,1,28,28)
        x = x.view(x.size(0), -1) 
        print('I am here',x.shape)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.dropout(x, 0.5)
        x = self.fc1(x)
        x = self.fc1relu(x)
        x = self.fc2(x)
        x = self.fc2relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.act(x)
       # x = torch.sigmoid(self.fc3(x))
        return x


net = ConvNet()

optimizer = optim.Adam(net.parameters(), lr=0.03)

loss_function = nn.BCELoss()

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)

epochs = 10
steps = 0
train_losses, test_losses = [], []
for e in range(epochs):
    running_loss = 0
    for images, labels in train_loader: 
        optimizer.zero_grad()
        log_ps = net(images)
        loss = loss_function(log_ps, labels)
        loss.backward()
        optimizer.step()        
        running_loss += loss.item()        
    else:
        test_loss = 0
        accuracy = 0        

        with torch.no_grad():
            for images, labels in test_loader: 
                log_ps = net(images)
                test_loss += loss_function(log_ps, labels)                
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class.type('torch.LongTensor') == labels.type(torch.LongTensor).view(*top_class.shape)
                accuracy += torch.mean(equals.type('torch.FloatTensor'))
        train_losses.append(running_loss/len(train_loader))
        test_losses.append(test_loss/len(test_loader))
        print("[Epoch: {}/{}] ".format(e+1, epochs),
              "[Training Loss: {:.3f}] ".format(running_loss/len(train_loader)),
              "[Test Loss: {:.3f}] ".format(test_loss/len(test_loader)),
              "[Test Accuracy: {:.3f}]".format(accuracy/len(test_loader)))