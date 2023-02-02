import os
import cv2
import random
import numpy as np
from numpy.lib.stride_tricks import as_strided

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import matplotlib.pyplot as plt

ifrom tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Activation
from tensorflow.keras.utils import to_categorical

data_set = []
data_label = []

training_data = []

input_path = "brain_tumor_dataset/"

CATEGORIES = ["no", "yes"]


def create_training_data():
    
    for category in CATEGORIES:
        path = os.path.join(input_path, category)
        category_index = CATEGORIES.index(category)
        for image in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
                img_array = img_array.astype(np.float32)

                  #Please ignore this line
#                 img_array = pool2d(img_array, kernel_size=2, stride=2, padding=0)

                img_array = cv2.resize(img_array, (256, 256))
                training_data.append([img_array, category_index])
            except Exception as e:
                print(e)
                
create_training_data()

random.shuffle(training_data)

for feature, label in training_data:
    data_set.append(feature)
    data_label.append(label)

x_train, x_test, y_train, y_test = train_test_split(data_set, data_label, test_size = 0.1, 
                                                    random_state = 45)

data_set = np.array(x_train).reshape(-1, 256, 256, 1)

x_test = np.array(x_test).reshape(-1, 256, 256, 1)

data_set = data_set/255.0

model = Sequential()
model.add(Conv2D(128, (3,3), input_shape = data_set.shape[1:]))
model.add(Activation("relu"))
model.add(AveragePooling2D(pool_size = (2,2)))

model.add(Conv2D(128, (3,3)))
model.add(Activation("relu"))
model.add(AveragePooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation("softmax"))

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])

model.fit(data_set, y_train, batch_size = 32, epochs = 15, validation_split = 0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
    
