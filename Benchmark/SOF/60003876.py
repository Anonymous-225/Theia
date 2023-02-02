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
import os
import pandas as pd 
#import matplotlib.pyplot as plt 

train_path = 'dogs-vs-cats/train/'
test_dir = os.listdir('dogs-vs-cats/test1/')
training_data = pd.DataFrame({'file': os.listdir('dogs-vs-cats/train/')})
#file_names = os.listdir(train_path)
labels = []
binary_labels = []
for path in os.listdir('dogs-vs-cats/train/'):
   if 'dog' in path:
        #labels.append('dog')
        binary_labels.append(1)
   else:
        #labels.append('cat')
        binary_labels.append(0)

#training_data['labels'] = labels
training_data['binary_labels'] = binary_labels
print(training_data.head())
test = pd.DataFrame({'file': os.listdir('dogs-vs-cats/test1/')})
print('This is test data',test.head())
# train_set, val_set = train_test_split(train,
#                                      test_size=0.2,random_state=42)
# print(len(train_set), len(val_set))


# def load_data(path):
# data = []
# ant = 0
# bee = 0
# for folder in os.listdir(path):
#     print(folder)
#     curfolder = os.path.join(path, folder)
#     for file in os.listdir(curfolder):
#         image = plt.imread(curfolder+'/'+file)
#         image = cv2.resize(image, (500,500))
#         if folder == 'ants':
#                 ant += 1
#                 data.append([np.array(image) , np.eye(2)[0]])
#         elif folder == 'bees':
#                 bee += 1
#                 data.append([np.array(image) , np.eye(2)[1]])

# np.random.shuffle(data)      
# np.save('train.npy',data)
# print('ants : ',ant)
# print('bees : ',bee)

# # training_data = np.load("train.npy",allow_pickle=True)
print('Length is',len(training_data))

class Net(nn.Module):
    def __init__(self):
        super().__init__() # just run the init of parent class (nn.Module)
        self.conv1 = nn.Conv2d(3, 32, 5) # input is 1 image, 32 output channels, 5x5 kernel / window
        self.conv2 = nn.Conv2d(32, 64, 5) # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 kernel / window
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(3,500,500).view(-1,3,500,500)
        self._to_linear = None
        self.convs(x)
        print(self._to_linear)
        self.fc1 = nn.Linear(self._to_linear, 512) #flattening.
        self.fc2 = nn.Linear(512, 2) # 512 in, 2 out bc we're doing 2 classes (dog vs cat).

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # bc this is our output layer. No activation here.
        return F.softmax(x, dim=1)

net = Net()

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()


train_X = torch.Tensor([i[0] for i in training_data]).view(-1,3,500,500)
train_X = train_X/255.0
train_y = torch.Tensor([i[1] for i in training_data])

device = torch.device("cuda:0")
net = Net().to(device)
print(len(train_X))
epochs = 10
BATCH_SIZE = 1
for epoch in range(epochs):
        for i in range(0, len(train_X), BATCH_SIZE): # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
            #print(f"{i}:{i+BATCH_SIZE}")
            batch_X = train_X[i:i+BATCH_SIZE]
            batch_y = train_y[i:i+BATCH_SIZE]

            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            net.zero_grad()
            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()     # Does the update

        print(f"Epoch : {epoch}. Loss: {loss}")