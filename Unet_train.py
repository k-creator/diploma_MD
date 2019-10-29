import numpy as np
import matplotlib.pyplot as plt
from Unet import Unet
import warnings
from keras.layers import Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras.backend as K
from tqdm import tqdm
import warnings
import os
import sys
import numpy as np


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 1

TRAIN_IMAGES = r'D:\CANCER\ready_dataset\images'
TRAIN_LABELS = r'D:\CANCER\ready_dataset\labels'

train_images = (next(os.walk(TRAIN_IMAGES))[2])[:300]
train_labels = (next(os.walk(TRAIN_LABELS))[2])[:300]

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

warnings.filterwarnings('ignore', category=UserWarning)
# print('Loading dataset...')
# X_train = np.load(r'D:\CANCER\ready_dataset\images_unet.npy')
# Y_train = np.load(r'D:\CANCER\ready_dataset\labels_unet.npy')

model = Unet()
input_img = Input((512, 512, 1), name='img')
unet_model = model.get_unet(input_img, n_filters=8, dropout=0.05, batchnorm=True)
unet_model.compile(optimizer='adam', loss=dice_coef_loss, metrics=['accuracy', dice_coef])

callbacks = [
    EarlyStopping(patience=7, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint(r'D:\CANCER\ready_dataset\lungs_cancer_seg.h5', verbose=1, save_best_only=True)
]

results = unet_model.fit(X_train, Y_train, validation_split=0.1, batch_size=1,
                         epochs=10, callbacks=callbacks)


# fig = plt.figure()
#
# ax_1 = fig.add_subplot(121)
# ax_1.imshow(np.reshape(X_train[0], (512, 512)), cmap='gray')
#
# ax_2 = fig.add_subplot(122)
# ax_2.imshow(np.reshape(Y_train[0] * 255, (512, 512)), cmap='gray')
# plt.show()
