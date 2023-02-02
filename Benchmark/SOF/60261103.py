from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K


sz=224 # image width = height = 224
batch_size=64
train_data_dir = "crack_dataset/train"
validation_data_dir = "crack_dataset/validate"
nb_train_samples = 3416
nb_val_samples = 612

train_datagen = ImageDataGenerator(rescale=1./255)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size = (sz, sz),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                              target_size = (sz, sz),
                                                              batch_size=batch_size,
                                                              class_mode='binary')

# Create Model 
model = Sequential()

model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11,11), strides=(4,4), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))

model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))

model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

model.add(Flatten())
model.add(Dense(4096, input_shape=(256,), activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(0.1), loss='binary_crossentropy', metrics=['accuracy'])

model.fit_generator(train_generator,
                    steps_per_epoch = nb_train_samples // batch_size, 
                    epochs=30,
                    validation_data=validation_generator,
                    validation_steps=nb_val_samples // batch_size)