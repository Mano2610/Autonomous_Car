"""
CNN for Steering - Experiment 1
-------------------------
Created on Wed Apr  7 08:35:21 2021

@author: kevin machado gamboa
"""
# -----------------------------------------------------------------------------
#                           Libraries Needed
# -----------------------------------------------------------------------------
import os
import cv2 as cv
import pandas as pd
import numpy as np
from tqdm import tqdm
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

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization

#%%
# -----------------------------------------------------------------------------
#                                  Data Loading
# -----------------------------------------------------------------------------
# Loading Labels
#project_path = os.getcwd()
project_path = 'D:\Folder\Essex\CE903-group _project\WindowsNoEditor\PythonAPI\ce903_team06'
info_data = pd.read_csv(project_path + '\data\_controls.csv', header=None)
# Filtering output variables (steering, throttle, break)
num_img = 7000
im_id = info_data.iloc[:num_img,0]
y = info_data.iloc[:num_img, 1:].to_numpy()
#%%
# Loading Images
newsize = (160, 80)
im_dataset_pil = []
X = []
for idx, name in tqdm(enumerate(im_id)):
    im_path = os.path.join(project_path+'\\data\\lane_detection\\compressed', (str(name)+'.jpg'))
    X.append(cv.resize(cv.imread(im_path), newsize))
    # X.append(np.array(Image.open(im_path).resize(newsize))/127.5-1.0)
#%%
# -----------------------------------------------------------------------------
#                                  Data Split
# -----------------------------------------------------------------------------
X = np.array(X)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

input_shape = X_train[0].shape
print('Input shape:', input_shape)
#%%
# -----------------------------------------------------------------------------
#                                  Loading Model
# -----------------------------------------------------------------------------
lane_model = tf.keras.models.load_model('full_CNN_model.h5')
lane_model.summary()
# Freeze lane model params
for layer in lane_model.layers:
    layer.trainable = True

base = lane_model.output
base = Flatten()(base)

# Steering output
steer = Dense(1000, activation='relu')(base)
steer = Dense(500, activation='relu')(steer)
steer = Dense(1, activation='tanh', name='steering_output')(steer)
# Throttle output
throttle = Dense(1000, activation='relu')(base)
throttle = Dense(500, activation='relu')(throttle)
throttle = Dense(1, activation='sigmoid', name='throttle_output')(throttle)
# Break output
brake = Dense(1000, activation='relu')(base)
brake = Dense(500, activation='relu')(brake)
brake = Dense(1, activation='sigmoid', name='break_output')(brake)

model = tf.keras.models.Model(
    inputs=lane_model.input,
    outputs=[steer,throttle,brake],
    name='transfer_learning'
)


# Model Parameters
optimizer_f = tf.keras.optimizers.Adam(lr=1.0e-3)
loss_f = tf.keras.losses.MeanSquaredError()
# Compiling model
model.compile(optimizer=optimizer_f,
                   loss=loss_f,
                   metrics='mse'
                    )

checkpoint_path, to_monitor = 'saved_model_full_data/', 'trainable_model-{epoch:03d}-{val_loss:.4f}'
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path+to_monitor+'.h5',
                                                monitor='val_loss',
                                                verbose=0,
                                                save_best_only=True,
                                                mode='auto')

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')
#%%
history_1 = model.fit(X_train, y_train, epochs=600, batch_size=50, validation_data=(X_val, y_val), callbacks = [early_stop, checkpoint])

mp1 = 'saved_model_full_data/transfer_model.png'
tf.keras.utils.plot_model(model, to_file=mp1, show_shapes=True)