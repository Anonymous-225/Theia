# Localizing Performance Bugs in Convolutional Neural Networks

Theia can be used to detect bugs in Keras programs and PyTorch Programs.

For Keras, you need to add class Theia in Theia_Keras.py as a subclass in your keras callbacks.py file.

Then you can pass Theia() to .fit() method of the model as follows:

```python
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape = X_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(10)) 
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
callback = Theia(x_train, x_test, batch_size, classes, input_type) #input_type = 1 for data augmentation
model_fit = model.fit(X_train, y_train, batch_size=32, epochs=30, validation_split=0.1,
                        callbacks=[callback])
```

For PyTorch, you need to save Theia_PyTorch.py file in the same folder as of buggy program.

Use Theia.check(train_data, test_data, model, loss, optimizer, batch_size) at the beginning of the training loop.

# Prerequisites
```
Python 3.7.16
Keras 2.3.0
Tensorflow 2.1.0
Torch 1.13.1
numpy 1.21.6
```
