import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers

import os
# dimensions of our images.
img_width, img_height = 512,512
train_data_dir = 'birds/train'
validation_data_dir = 'birds/test'

epochs = 20
batch_size = 64

input_shape = (img_width, img_height, 3)
reg = 0.0001

model = Sequential()

model.add(Conv2D(8, (3, 3), input_shape=input_shape, padding='same',
        kernel_regularizer=regularizers.l2(reg)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.7))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(16, (3, 3), input_shape=input_shape, padding='same',
        kernel_regularizer=regularizers.l2(reg)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.7))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(16, kernel_regularizer=regularizers.l2(reg)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(6))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='SGD',metrics=['accuracy'])
train_datagen = ImageDataGenerator(rotation_range=10,
                                width_shift_range=0.05,
                                height_shift_range=0.05,
                                shear_range=0.05,
                                zoom_range=0.05,
                                rescale=1/255.,
                                fill_mode='nearest',
                                channel_shift_range=0.2*255)
train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            shuffle = True,
            class_mode='categorical')

validation_datagen = ImageDataGenerator(rescale=1/255.)
validation_generator = validation_datagen.flow_from_directory(
                                        validation_data_dir,
                                        target_size=(img_width, img_height),
                                        batch_size=1,
                                        shuffle = True,
                                      class_mode='categorical')
history = model.fit_generator(train_generator,
                          epochs=20,
                          validation_data=validation_generator))