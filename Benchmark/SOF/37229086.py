from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Activation
from tensorflow.keras.utils import to_categorical


batch_size = 32
epoch = 30
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

model = Sequential()
model.add(Conv2D(256, (3, 3), padding='same',activation='relu', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(128, (3, 3), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Flatten())
model.add(Dense(384,activation='relu'))
model.add(Dense(192,activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

cnn = model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch, validation_data=(x_test,y_test)
                ,shuffle=True, verbose= 0)
