

from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

batch_size=128
epoch=30
img_rows, img_col = 32,32

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape(x_train.shape[0],32,32,3)
x_test = x_test.reshape(x_test.shape[0],32,32,3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


model4 = Sequential()
model4.add(Conv2D(32, (3,3), activation='relu',padding='same', kernel_initializer='he_uniform', 
                  input_shape=(32, 32, 3)))
model4.add(MaxPooling2D((2, 2)))
model4.add(Dropout(0.2))
model4.add(Conv2D(64, (3, 3),padding='same', activation='relu'))
model4.add(MaxPooling2D((2, 2)))
model4.add(Dropout(0.2))

model4.add(Flatten())
model4.add(Dense(128, activation='relu'))
model4.add(Dropout(0.5))
model4.add(Dense(64, activation='relu'))
model4.add(Dropout(0.5))
model4.add(Dense(32, activation='relu'))
model4.add(Dropout(0.5))
model4.add(Dense(16, activation='relu'))
model4.add(Dropout(0.5))
model4.add(Dense(10, activation='softmax'))

opt = keras.optimizers.Adam(learning_rate = 0.001)
model4.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model4.fit(x_train,y_train,
          batch_size=batch_size,
          epochs=epoch,
          verbose=1,
          validation_data=(x_test, y_test))

loss, accuracy = model4.evaluate(x_test, y_test)
print('loss: ', loss, '\naccuracy: ', accuracy)
