
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K

import os
# dimensions of our images.
img_width, img_height = 75,75

train_path = 'flowers/train'
classes = os.listdir(train_path)
test_path = 'flowers/test'
epochs = 20
batch_size = 64

input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(16,(2,2),activation='relu',input_shape=(75,75,3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(16,(2,2),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(32,(2,2),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(5,activation='sigmoid'))

model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.0001),
             loss='categorical_crossentropy',
             metrics=['acc'])

# this is the augmentation configuration we will use for training
train_gen = ImageDataGenerator(
 rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_gen = ImageDataGenerator(rescale=1./255)  
  
train_data = train_gen.flow_from_directory(
    train_path, 
    target_size=(img_width, img_height),
    batch_size = 64, 
    class_mode = "categorical" ,
    classes  = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'],
    shuffle = True
)

test_data = test_gen.flow_from_directory(
    test_path, 
    target_size=(img_width, img_height),
    batch_size = 64, 
    class_mode = "categorical" ,
    shuffle = True
)

history = model.fit_generator(train_data,
                              epochs=20,
                              validation_data=test_data))

