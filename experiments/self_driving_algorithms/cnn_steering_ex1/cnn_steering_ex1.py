"""
CNN for Steering
-----------------
Created on Sat Mar  6 23:48:21 2021

@author: kevin machado gamboa
"""
# -----------------------------------------------------------------------------
#                                Libraries
# -----------------------------------------------------------------------------
import os
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
from tensorflow.keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
#im = image.load_img(im_list[0])

#%%
# Renaming image files - Necessary for Tensorflow Image Generator
# im_path = os.path.abspath('..\..\..\data\lane_detection')
# im_files = os.listdir(im_path)
# for index, name in enumerate(im_files):
#   os.rename(os.path.join(im_path, name), os.path.join(im_path, 'im'+name))
#%%
# Loading Labels
data_ann_path = '../../../data'
data_out = pd.read_csv(data_ann_path + '/_controls.csv')
y = data_out.steer
# Loading Images
im_path = os.path.abspath('..\..\..\data\lane_detection')
im_files = os.listdir(im_path)
newsize = (300, 300)
im_dataset_pil = []
X = []
for idx, name in tqdm(enumerate(im_files)):
  im_dataset_pil.append(Image.open(os.path.join(im_path, name)))
  X.append(np.array(im_dataset_pil[idx].resize(newsize)))

# im_dataset_pil[0].show()
# plt.show()
#%%
# Data split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
X_train = np.expand_dims(X_train, axis=3)
X_val = np.expand_dims(X_val, axis=3)
#%%
# Ref: https://github.com/llSourcell/How_to_simulate_a_self_driving_car/blob/master/model.py
input_shape = X_train[0].shape


def model_steering():
  model = tf.keras.Sequential([
    Conv2D(24, 5, 5, activation='elu', padding='same', input_shape=input_shape),
    Conv2D(36, 5, 5, activation='elu', padding='same'),
    Conv2D(48, 5, 5, activation='elu', padding='same'),
    Conv2D(64, 3, 3, activation='elu', padding='same'),
    Conv2D(64, 3, 3, activation='elu', padding='same'),
    Dropout(0.5),
    Flatten(),
    Dense(100, activation='elu'),
    Dense(50, activation='elu'),
    Dense(10, activation='elu'),
    Dense(1)
  ])
  # Model Parameters
  model.compile(optimizer='adam',
                loss='mse',
                metrics=['accuracy'])
  return model

model = model_steering()
model.summary()
#%%
checkpoint_path = 'saved_model/'
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path+'model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')
X_train, X_val, y_train, y_val
history_1 = model.fit(X_train, y_train, epochs=600, batch_size=100, validation_data=(X_val, y_val), callbacks = [early_stop])

# img = Image.open(str(im_list[0]))
# # Loading Images
# im_dir = Path(im_path)
# im_list = list(im_dir.glob('**/*.png'))
# print('%i images found' %len(im_list))
# #%%
# im_path = os.path.abspath('../../../data/lane_detection')
# data_ann_path = '../../../data'
# data_out = pd.read_csv(data_ann_path + '/_controls.csv')
# im_dir = Path(im_path)
# im_list = list(im_dir.glob('**/*.png'))
# print('%i images found' %len(im_list))

#im_data = os.listdir(im_path)


#%%
# batch_size = 32
# image_size = 254, 254
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   im_dir,
#   seed=123,
#   image_size=image_size,
#   batch_size=batch_size)

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   im_path,
#   validation_split=0.2,
#   subset="validation",
#   seed=123,
#   image_size=image_size,
#   batch_size=batch_size)
#%%


# len(list(im_dir.glob('*/*.png')))


fig, ax = plt.subplots(1, 1)
ax.set_aspect('equal')

x = data_out.steer
y = np.sqrt(-x**2. + 0.6)

