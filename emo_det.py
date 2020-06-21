import tensorflow as tf
import keras
import pandas as pd
import numpy as np


csv_reader = pd.read_csv('fer2013.csv')
# Test read the csv headings with Pandas
# print(csv_reader.head())

TrainX = []
TrainY = []
TestX = []
TestY = []

# Refer the csv file with MS Excel for the headings!
for index, row in csv_reader.iterrows():
    pixel = row['pixels'].split(" ")

    try:
        if 'Training' in row['Usage']:
            TrainX.append(np.array(pixel, 'float32'))
            TrainY.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
            TestX.append(np.array(pixel, 'float32'))
            TestY.append(row['emotion'])
    except:
        print(f"Error @ index: {index}, row: {row}")

# Test print the data!
# print(TrainX[0:2])
# print(TrainY[0:2])

TrainX = np.array(TrainX, 'float32')
TrainY = np.array(TrainY, 'float32')
TestX = np.array(TestX, 'float32')
TestY = np.array(TestY, 'float32')

# Normalizing data
# Values should be between 0 and 1
# Training Data
TrainX = TrainX - np.mean(TrainX, axis=0)
TrainX = TrainX / np.std(TrainX, axis=0)

# Testing Data
TestX = TestX - np.mean(TestX, axis=0)
TestY = TestY / np.std(TestY, axis=0)

# Kaggle Data set contains grayscale images of shape (48, 48)
TrainX = TrainX.reshape(TrainX.shape[0], 48, 48, 1)
TestX = TestX.reshape(TestX.shape[0], 48, 48, 1)
TrainY = tf.keras.utils.to_categorical(TrainY, num_classes=7)
TestY = tf.keras.utils.to_categorical(TestY, num_classes=7)

model = tf.keras.Sequential([
    # First Conv Layer
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                           input_shape=(TrainX.shape[1:])),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Second Conv Layer
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Third Conv Layer
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.7),
    tf.keras.layers.Dense(7, activation='softmax'),
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# model.summary()

model.fit(
    TrainX,
    TrainY,
    batch_size=64,
    epochs=50,
    verbose=1,
    shuffle=True,
    validation_data=(TestX, TestY)
)

json_model = model.to_json()
with open("saved/saved_model.json", "w") as json_file:
    json_file.write(json_model)

model.save_weights("saved/saved_weights.h5")
