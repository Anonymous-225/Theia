
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K

import os
# dimensions of our images.
img_width, img_height = 150,150
train_data_dir = 'birds_3/train'
validation_data_dir = 'birds_3/test'

epochs = 20
batch_size = 32

input_shape = (img_width, img_height, 3)
node_size=64

model = Sequential()
model.add(Conv2D(64,(3,3),input_shape = input_shape))
#idk what that shape does except that and validation i have no problem 
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(3))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])



train_datagen = ImageDataGenerator(rescale=1./255,
                                      shear_range=0.2,
                                         zoom_range=0.2,
                                         horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            shuffle = True,
            class_mode='categorical')

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
                                        validation_data_dir,
                                        target_size=(img_width, img_height),
                                        shuffle = True,
                                        batch_size=batch_size,
                                       
                                      class_mode='categorical')

history = model.fit_generator(train_generator,
                              validation_data=validation_generator,
                          epochs=20,
                          )

