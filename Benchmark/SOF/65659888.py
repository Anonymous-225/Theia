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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Hl_Model(nn.Module):

    torch.cuda.set_device(0)

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2)

        x = torch.randn(145,200).view(-1,1,145,200)
        self._to_linear = None
        self.convs(x)
    
        self.fc1 = nn.Linear(self._to_linear, 128, bias=True)
        self.fc2 = nn.Linear(128, 3)

    def convs(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2), stride=2)

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x


    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
optimizer = optim.Adam(net.parameters(), lr=0.05)
loss_fun = nn.CrossEntropyLoss()


def train(net, train_fold_x, train_fold_y):
    
  
    BATCH_SIZE = 5
    EPOCHS = 50
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_fold_x), BATCH_SIZE)):

            batch_x = train_fold_x[i:i+BATCH_SIZE].view(-1, 1, 145, 200)
            batch_y = train_fold_y[i:i+BATCH_SIZE]
        
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
            optimizer.zero_grad()
            outputs = net(batch_x)

            batch_y = batch_y.long()
            loss = loss_func(outputs, batch_y)

            loss.backward()
            optimizer.step()
        
        print(f"Epoch: {epoch} Loss: {loss}")
def test(net, test_fold_x, test_fold_y):
    
    test_fold_x.to(device)
    test_fold_y.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_fold_x))):
            real_class = torch.argmax(test_fold_y[i]).to(device)
            net_out = net(test_fold_x[i].view(-1, 1, 145, 200).to(device))
            pred_class = torch.argmax(net_out)

            if pred_class == real_class:
                correct += 1
            total +=1