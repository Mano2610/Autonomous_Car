'''
Lane Detection Experiment 1
# -------------------------
Created on Thu Feb 25 21:51:33 2021

@author: Kevin Machado Gamboa

References:
    1. https://github.com/mvirgo/MLND-Capstone
'''
# -----------------------------------------------------------------------------
#                                Libraries
# -----------------------------------------------------------------------------
import os
import cv2 as cv
import numpy as np
#from scipy.misc import imresize
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
# %%
# -----------------------------------------------------------------------------
#                          Functions Implemented
# -----------------------------------------------------------------------------

def predict_lane(image, model):
    # Image shape
    height, width = image.shape[0], image.shape[1]
    small_img = cv.resize(image, (80, 160))
    small_img = np.array(small_img)
    small_img = small_img[None, :, :, :]
    prediction = model.predict(small_img)[0] * 255

    k = []
    k.append(prediction)
    # Results pre-process
    kavg = np.mean(np.array([i for i in k]), axis=0)
    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(kavg).astype(np.uint8)
    # creates RGB image with prediction located in G
    lane_drawn = np.dstack((blanks, kavg, blanks))
    # Re-size to match the original image
    lane_image = cv.resize(np.uint8(lane_drawn), (width, height))
    # Merge the lane drawing onto the original image
    #image = cv.addWeighted(image, 0.5, lane_image, 0.5, 0)
    return cv.addWeighted(image, 0.5, lane_image, 0.5, 0)
# %%
# -----------------------------------------------------------------------------
#                          Loading Images
# -----------------------------------------------------------------------------
im_folder = '../lane_dataset'
images = [cv.imread(os.path.join(im_folder, image)) for image in os.listdir(im_folder)]

im = images[:17]
#or_size = (im.shape[0], im.shape[1])
fig, axes = plt.subplots(nrows=17, ncols=2, figsize=(10,20))
axes[0,0].set_title('Original')
for i in range(17):
    axes[i,0].imshow(im[i], aspect='auto')
# plt.show()
# %%
# -----------------------------------------------------------------------------
#                          Loading TFLite model
# -----------------------------------------------------------------------------
#
model = load_model('full_CNN_model.h5')
model_insize = model.input_shape
# --------------------------
dot_img_file = 'model_1.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
# -------------------------

# %%
# -----------------------------------------------------------------------------
#                          Model Predictions
# -----------------------------------------------------------------------------
for i in range(17):
    image = predict_lane(im[i], model)
    axes[i,1].imshow(image, aspect='auto')

axes[0,1].set_title('Prediction')
plt.show()