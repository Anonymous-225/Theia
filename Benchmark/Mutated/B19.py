
from tensorflow.keras.models import Sequential
import  tensorflow 
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

batch_size = 64
epoch = 30
img_rows, img_col = 150,150

train_path = 'flowers/train'
test_path = 'flowers/test'
classes = os.listdir(train_path)
for folder in classes:
    print(folder)
train_gen = ImageDataGenerator(
                        rescale=1./255,
                        rotation_range=30,
                         width_shift_range=0.1, height_shift_range=0.1,
                          shear_range=0.2, zoom_range=0.2,
                           horizontal_flip=True, fill_mode='nearest')
test_gen = ImageDataGenerator(rescale=1./255)
    
train_data = train_gen.flow_from_directory(
    train_path, 
    target_size=(150,150),
    batch_size = 64, 
    class_mode = "categorical" ,
    classes  = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'],
    shuffle = True
)

test_data = test_gen.flow_from_directory(
    test_path, 
    target_size=(150,150),
    batch_size = 64, 
    class_mode = "categorical" ,
    shuffle = True
)
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'same',activation ='relu', input_shape = (150,150,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
 

model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(5, activation = "softmax"))

opt = tensorflow.keras.optimizers.Adam(learning_rate = 0.000001)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])


history = model.fit(train_data, epochs = 30, validation_data=test_data)
                    
model.evaluate(test_data)