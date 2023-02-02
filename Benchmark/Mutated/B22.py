


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
import os
import pandas as pd 
from sklearn.model_selection import train_test_split


batch_size =128
img_row,img_column = 224,224
epoch =20
train_path = 'dogs-vs-cats/train/'
test_dir = os.listdir('dogs-vs-cats/test1/')
train = pd.DataFrame({'file': os.listdir('dogs-vs-cats/train/')})

labels = []
binary_labels = []
for path in os.listdir('dogs-vs-cats/train/'):
   if 'dog' in path:
        labels.append('dog')
        binary_labels.append(1)
   else:
        labels.append('cat')
        binary_labels.append(0)

train['labels'] = labels
train['binary_labels'] = binary_labels
print(train.head())
test = pd.DataFrame({'file': os.listdir('dogs-vs-cats/test1/')})
train_set, val_set = train_test_split(train,
                                     test_size=0.2,random_state=42)
print(len(train_set), len(val_set))

datagen_train = ImageDataGenerator(rescale=1./255,
                          shear_range=0.2, zoom_range=0.2,
                           horizontal_flip=True)

# Augment validating data
datagen_valid = ImageDataGenerator(rescale=1./255)

aug_train = datagen_train.flow_from_dataframe(train_set, directory=train_path, 
                                             x_col='file', y_col='labels',
                                             target_size=(224,224), class_mode='categorical',
                                             batch_size=batch_size)

aug_test = datagen_valid.flow_from_dataframe(val_set, directory=train_path,
                                             x_col='file', y_col='labels',
                                             target_size=(224,224), class_mode='categorical',
                                             batch_size=batch_size)



model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(224,224,3),padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

hist = model.fit(aug_train, validation_data=aug_test, epochs=epoch, verbose=1)
                




