import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Created by Ebubekir Kocaman
# reference: https://www.pyimagesearch.com/2019/11/04/traffic-sign-classification-with-keras-and-deep-learning/

"""
This code was done as experiment of CNN model
"""

class TrafficSign:
    def __init__(self, data_path):
        self.data_path = data_path
        pass

    def get_data(self, dataset):
        images = []
        classes = []
        rows = pd.read_csv(dataset)
        rows = rows.sample(frac=1).reset_index(drop=True)
        for i, row in rows.iterrows():
            img_class = row["ClassId"]
            img_path = row["Path"]
            image = os.path.join(self.data_path, img_path)
            image = cv2.imread(image)
            image_rs = cv2.resize(image, (32, 32), 3)
            R, G, B = cv2.split(image_rs)
            img_r = cv2.equalizeHist(R)
            img_g = cv2.equalizeHist(G)
            img_b = cv2.equalizeHist(B)
            new_image = cv2.merge((img_r, img_g, img_b))
            if i % 500 == 0:
                print(f"loaded: {i}")
            images.append(new_image)
            classes.append(img_class)
        X = np.array(images)
        y = np.array(images)

        return (X, y)

    def createCNN(self, width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        model.add(Conv2D(8, (5, 5), input_shape=inputShape, activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(16, (3, 3), activation="relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(16, (3, 3), activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(512, activation="relu"))
        model.add(Dense(classes, activation="softmax"))
        return model