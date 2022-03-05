"""
Created by Ebubekir Kocaman

Reference for code: https://towardsdatascience.com/traffic-sign-recognition-using-deep-neural-networks-6abdb51d8b70
Dataset source: https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

from sklearn.metrics import accuracy_score

from image_classes import classes

class TrafficSign:
    def __init__(self, data, labels, classes, epochs):
        self.path = "../../data/traffic_sign_detection"
        self.data = data
        self.labels = labels
        self.classes = classes
        self.epochs = epochs
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_train = None
        self.model = None
        self.results = None

    def load_data(self):
        for i in range(self.classes):
            img_path = os.path.join(f'{self.path}/train', str(i))
            images = os.listdir(img_path)

            for j in images:
                try:
                    img = Image.open(f"{img_path}/{j}")
                    # image resizing to reduce size
                    img_resize = img.resize((30, 30))
                    np_img = np.array(img_resize)
                    self.data.append(np_img)
                    self.labels.append(i)
                except:
                    print("Error getting image")

    def preprocess_data(self):
        # numpy arrays conversion for faster speed and less RAM usage
        data = np.array(self.data)
        labels = np.array(self.labels)

        print(data.shape, labels.shape)

        # data split for train and test set
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=68)

        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        self.x_train = x_train
        self.x_test = x_test

        # one hot encoding conversion
        self.y_train = to_categorical(y_train, 43)
        self.y_test = to_categorical(y_test, 43)

        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    def create_cnn(self):
        # CNN model creation
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=self.x_train.shape[1:]))
        model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(43, activation='softmax'))

        self.model = model

    def train(self):
        # model training
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.results = self.model.fit(self.x_train, self.y_train, batch_size=32, epochs=self.epochs,
                                 validation_data=(self.x_test, self.y_test))

        # saving model to load with re-training
        self.model.save("traffic_signs_model.h5")

    def visualise_accuracy(self):
        # plot graphs for accuracy measure
        plt.figure(0)
        plt.plot(self.results.history['accuracy'], label='training accuracy')
        plt.plot(self.results.history['val_accuracy'], label='val accuracy')
        plt.title('Accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()

    def visualise_loss(self):
        # plot graphs for loss measure
        plt.figure(1)
        plt.plot(self.results.history['loss'], label='training loss')
        plt.plot(self.results.history['val_loss'], label='val loss')
        plt.title('Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

    def test(self):
        y_test = pd.read_csv(f'{self.path}/test.csv')

        labels = y_test["ClassId"].values
        imgs = y_test["Path"].values

        data = []

        for img in imgs:
            image = Image.open(img)
            image = image.resize((30, 30))
            data.append(np.array(image))

        x_test = np.array(data)

        pred = self.model.predict_classes(x_test)

        # Accuracy with the test data
        print(accuracy_score(labels, pred))

    def run_model(self):
        self.load_data()
        self.preprocess_data()
        self.create_cnn()
        self.train()
        self.visualise_accuracy()
        self.visualise_loss()
        self.test()


def classify_images(image_path, images):
    # model loading
    model = load_model("traffic_signs_model.h5")

    for i in range(len(images)):
        img_path = f"{image_path}/{images[i]}"

        # image preprocess for prediction
        img = Image.open(img_path)
        img_resized = img.resize((30, 30))
        img_dim = np.expand_dims(img_resized, axis=0)
        np_img = np.array(img_dim)

        # image prediction from model
        pred = model.predict_classes([np_img])[0]

        img_class = classes[pred+1]

        # image display with label as the class
        plt_img = image.load_img(img_path, target_size=(200, 200))
        plt.imshow(plt_img, cmap='Greys_r')
        plt.title(img_class)
        plt.axis('off')
        plt.show()
