import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Activation
from tensorflow.keras.utils import to_categorical


normalizedTrainingSet = ImageDataGenerator(rescale=1 / 255)
normalizedTestingSet = ImageDataGenerator(rescale=1 / 255)


trainingClass = normalizedTrainingSet.flow_from_directory("DataSet/Training",
                                                          target_size=(100, 100),
                                                          class_mode="categorical",
                                                          shuffle=True)

testingClass = normalizedTrainingSet.flow_from_directory("DataSet/Testing",
                                                         target_size=(100, 100),
                                                         class_mode="categorical",
                                                         shuffle=True)

print(trainingClass.class_indices)
print(testingClass.class_indices)

model = tf.keras.models.Sequential \
        ([
        tf.keras.layers.Conv2D(200, (3, 3), activation="relu", input_shape=(100, 100, 3)),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(200, (3, 3), activation="relu", input_shape=(100, 100, 3)),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(200, (3, 3), activation="relu", input_shape=(100, 100, 3)),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(200, (3, 3), activation="relu", input_shape=(100, 100, 3)),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(142, activation="softmax"),
        tf.keras.layers.Dense(7, activation="sigmoid")
    ])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

fittedModel = model.fit(trainingClass, epochs=1, validation_data=testingClass, shuffle=True)

