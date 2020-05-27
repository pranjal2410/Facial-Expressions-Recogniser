%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
import sys
import cv2 as cv
import pandas as pd

import tensorflow as tf
from keras.layers import Dense, Conv2D, MaxPool2D, AveragePooling2D, Input, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.utils import to_categorical
from keras.models import Sequential, Model, model_from_json
from keras.optimizers import adam, rmsprop

angry = os.listdir('Data/Train_set/Angry/')
happy = os.listdir('Data/Train_set/Happy/')
neutral = os.listdir('Data/Train_set/Neutral/')
sad = os.listdir('Data/Train_set/Sad/')
surprise = os.listdir('Data/Train_set/Surprise/')

angry_test = os.listdir('Data/validation/validation/Angry/')
happy_test = os.listdir('Data/validation/validation/Happy/')
neutral_test = os.listdir('Data/validation/validation/Neutral/')
sad_test = os.listdir('Data/validation/validation/Sad/')
surprise_test = os.listdir('Data/validation/validation/Surprise/')

file_path_train = 'Data/Train_set/'
file_path_test = 'Data/validation/validation/'

train_images = []
labels = []
test_images = []
test_labels = []

for i in angry:
    image = cv.imread(file_path_train + 'Angry/' + i)
    img = cv.resize(image, (48, 48))
    train_images.append(img)
    labels.append(0)
    
for i in angry_test:
    image = cv.imread(file_path_test + 'Angry/' + i)
    img = cv.resize(image, (48, 48))
    test_images.append(img)
    test_labels.append(0)

for i in happy:
    image = cv.imread(file_path_train + 'Happy/' + i)
    img = cv.resize(image, (48, 48))
    train_images.append(img)
    labels.append(1)
    
for i in happy_test:
    image = cv.imread(file_path_test + 'Happy/' + i)
    img = cv.resize(image, (48, 48))
    test_images.append(img)
    test_labels.append(1)

for i in sad:
    image = cv.imread(file_path_train + 'Sad/' + i)
    img = cv.resize(image, (48, 48))
    train_images.append(img)
    labels.append(2)
    
for i in sad_test:
    image = cv.imread(file_path_test + 'Sad/' + i)
    img = cv.resize(image, (48, 48))
    test_images.append(img)
    test_labels.append(2)

for i in neutral:
    image = cv.imread(file_path_train + 'Neutral/' + i)
    img = cv.resize(image, (48, 48))
    train_images.append(img)
    labels.append(3)
    
for i in neutral_test:
    image = cv.imread(file_path_test + 'Neutral/' + i)
    img = cv.resize(image, (48, 48))
    test_images.append(img)
    test_labels.append(3)

for i in surprise:
    image = cv.imread(file_path_train + 'Surprise/' + i)
    img = cv.resize(image, (48, 48))
    train_images.append(img)
    labels.append(4)
    
for i in surprise_test:
    image = cv.imread(file_path_test + 'Surprise/' + i)
    img = cv.resize(image, (48, 48))
    test_images.append(img)
    test_labels.append(4)

train_images = np.array(train_images, dtype='float') / 255.0
labels = np.array(labels)

test_images = np.array(test_images, dtype='float') / 255.0
test_labels = np.array(test_labels)

labels = to_categorical(labels, num_classes=5)
test_labels = to_categorical(test_labels, num_classes=5)

X_train, X_test, y_train, y_test = train_test_split(train_images, labels, test_size=0.2, random_state=0)

def build_model():
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu', input_shape=(48, 48, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(512, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(5, activation="softmax"))
    return model


classifier = build_model()
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier.summary()

H = classifier.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

preds = classifier.predict(test_images)
cm = confusion_matrix(test_labels.argmax(axis=1), preds.argmax(axis=1))
sum = np.trace(cm)
sum
acc = sum / test_images.shape[0]
acc

%matplotlib qt5
N = np.arange(0, 20)
plt.style.use('ggplot')
plt.figure(figsize=(12, 9))
plt.plot(N, H.history['loss'], label='train_loss')
plt.plot(N, H.history['val_loss'], label='val_loss')
plt.plot(N, H.history['accuracy'], label='train_accuracy')
plt.plot(N, H.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('loss/accuracy')
plt.title('Train vs Val (loss and accuracy)')
plt.legend()
plt.show()

classifier.save('models/modelv2.h5')
json_model = classifier.to_json()
with open('models/modelv2.json', 'w') as json_file:
    json_file.write(json_model)
    
classifier.save_weights('models/modelv2.weights')
    