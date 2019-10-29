from tqdm import tqdm
import warnings
import os
import sys
import random

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from itertools import chain

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 1

TRAIN_IMAGES = r'D:\CANCER\ready_dataset\images'
TRAIN_LABELS = r'D:\CANCER\ready_dataset\labels'

train_images = next(os.walk(TRAIN_IMAGES))[2]
train_labels = next(os.walk(TRAIN_LABELS))[2]

X_train = np.zeros((len(train_images), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
Y_train = np.zeros((len(train_labels), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
sys.stdout.flush()

for n, id_ in tqdm(enumerate(train_images), total=len(train_images)):
    path_image = TRAIN_IMAGES + r'\\' + id_
    img_npy = np.load(path_image)
    X_train[n] = np.reshape(img_npy, (512, 512, 1))

np.save(r'D:\CANCER\ready_dataset\images_unet.npy', X_train)
print('images loaded')

for n, id_ in tqdm(enumerate(train_labels), total=len(train_labels)):
    path_label = TRAIN_LABELS + r'\\' + id_
    label_npy = np.load(path_label)
    Y_train[n] = np.reshape(label_npy / 255, (512, 512, 1))

np.save(r'D:\CANCER\ready_dataset\labels_unet.npy', Y_train)
print('labels loaded')

