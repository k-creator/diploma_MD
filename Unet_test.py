from keras.models import load_model
import numpy as np
import os
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import keras.backend as K


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

TEST_IMAGES = r'D:\CANCER\ready_dataset\test_images'
TEST_LABELS = r'D:\CANCER\ready_dataset\test_labels'

test_images = (next(os.walk(TEST_IMAGES))[2])[:100]
test_labels = (next(os.walk(TEST_LABELS))[2])[:100]

X_test = np.zeros((len(test_images), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
Y_test = np.zeros((len(test_labels), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
sys.stdout.flush()

for n, id_ in tqdm(enumerate(test_images), total=len(test_images)):
    path_image = TEST_IMAGES + r'\\' + id_
    img_npy = np.load(path_image)
    X_test[n] = np.reshape(img_npy, (512, 512, 1))

for n, id_ in tqdm(enumerate(test_labels), total=len(test_labels)):
    path_label = TEST_LABELS + r'\\' + id_
    label_npy = np.load(path_label)
    Y_test[n] = np.reshape(label_npy, (512, 512, 1))

model = load_model(r'D:\CANCER\ready_dataset\models\f_16_m_dicecoef.h5',
                   custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})


for index, x_test in enumerate(X_test):
    pred = model.predict(np.reshape(x_test, (1, 512, 512, 1)))
    pred[pred > np.mean(pred)] = 255
    pred[pred < 255] = 0
    print(np.max(pred))
    fig = plt.figure()
    ax_1 = fig.add_subplot(131)
    ax_1.imshow(np.reshape(x_test, (512, 512)), cmap='gray')

    ax_2 = fig.add_subplot(132)
    ax_2.imshow(np.reshape(Y_test[index], (512, 512)), cmap='gray')

    ax_3 = fig.add_subplot(133)
    ax_3.imshow(np.reshape(pred, (512, 512)))

    plt.show()
