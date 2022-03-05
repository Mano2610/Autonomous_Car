'''
Car Detection Experiment 1
# -------------------------
Created on Thu Feb 20 16:35:45 2021

@author: Kevin Machado Gamboa

References:
    1. https://github.com/mvirgo/MLND-Capstone
'''
# -----------------------------------------------------------------------------
#                                Libraries
# -----------------------------------------------------------------------------
import os
import cv2
import matplotlib.pyplot as plt

import detector as dt
from visualization_utils import draw_bounding_boxes_on_image_array


# %%
# -----------------------------------------------------------------------------
#                          Loading Images
# -----------------------------------------------------------------------------
im_folder = '../objects_dataset'
images = [cv2.imread(os.path.join(im_folder, image)) for image in os.listdir(im_folder)]
im = images[:6]
#or_size = (im.shape[0], im.shape[1])
fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(10,20))
axes[0,0].set_title('Original')
for i in range(6):
    axes[i,0].imshow(im[i], aspect='auto')
# %%
# -----------------------------------------------------------------------------
#                          Loading TFLite model
# -----------------------------------------------------------------------------
#
model_path = 'detect.tflite'  # Tflite model path
label_path = 'labelmap.txt'  # model labels path
confidence = 0.5  # Minimum required confidence level of bounding boxes
#    args = parse_args()
detector = dt.ObjectDetectorLite(model_path=model_path, label_path=label_path)
input_size = detector.get_input_size()
# %%
# -----------------------------------------------------------------------------
#                          Model Predictions
# -----------------------------------------------------------------------------
for i in range(6):
    # resize image
    image = cv2.resize(im[i], tuple(input_size))  # reshape(im)  # Reshape the image
    # model predictions
    boxes, scores, classes = detector.detect(image, confidence)
    # plots the boxes if any
    if len(boxes) > 0:
        draw_bounding_boxes_on_image_array(image, boxes, display_str_list=classes)

    # %%
    axes[i,1].imshow(image, aspect='auto')
axes[0,1].set_title('Prediction')
plt.show()
