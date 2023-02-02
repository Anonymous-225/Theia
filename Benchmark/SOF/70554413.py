from tensorflow.keras.optimizers import SGD

X_train = X_train/255.0
model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape = X_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(16)) # added 16 because it model.fit gave error on 15 
model.add(Activation('softmax'))



model.compile(loss='sparse_categorical_crossentropy', 
             optimizer=SGD(learning_rate=0.01), 
             metrics=['accuracy'])

model_fit = model.fit(X_train, y_train, batch_size=32, epochs=30, validation_split=0.1)