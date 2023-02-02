from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Activation
from tensorflow.keras.utils import to_categorical

import cv2
import os

NUM_CLASSES = 5
IMG_SIZE = 150
img_rows, img_col = 150,150
train_path = 'flowers/train'
test_path = 'flowers/test'
classes = os.listdir(train_path)
for folder in classes:
    print(folder)
train_gen = ImageDataGenerator(
                        rotation_range=30,
                         width_shift_range=0.1, height_shift_range=0.1,
                          shear_range=0.2, zoom_range=0.2,
                           horizontal_flip=True, fill_mode='nearest')
test_gen = ImageDataGenerator() 
    
train_data = train_gen.flow_from_directory(
    train_path, 
    target_size=(150,150),
    batch_size = 16, 
    class_mode = "categorical" ,
    classes  = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'],
    shuffle = True
)

test_data = test_gen.flow_from_directory(
    test_path, 
    target_size=(150,150),
    batch_size = 16, 
    class_mode = "categorical" ,
    shuffle = True)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), strides=2, activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(Dropout(0.5))
model.add(Conv2D(128, kernel_size=(3, 3), strides=2, activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(128, kernel_size=(3, 3), strides=2, activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(128, kernel_size=(3, 3), strides=2, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_data, epochs=10, validation_data=test_data, verbose=1)