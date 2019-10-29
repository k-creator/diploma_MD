from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add


class Unet(object):
    def __init__(self):
        a = 0

    @staticmethod
    def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
        # first layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                   padding="same")(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
        # second layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                   padding="same")(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    def get_unet(self, input_img, n_filters=32, dropout=0.3, batchnorm=True):
        # contracting path
        kernel = 3
        c1 = self.conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=kernel, batchnorm=batchnorm)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout * 0.5)(p1)

        c2 = self.conv2d_block(p1, n_filters=n_filters * 2, kernel_size=kernel, batchnorm=batchnorm)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)

        c3 = self.conv2d_block(p2, n_filters=n_filters * 4, kernel_size=kernel, batchnorm=batchnorm)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)

        c4 = self.conv2d_block(p3, n_filters=n_filters * 8, kernel_size=kernel, batchnorm=batchnorm)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)
        p4 = Dropout(dropout)(p4)

        c5 = self.conv2d_block(p4, n_filters=n_filters * 16, kernel_size=kernel, batchnorm=batchnorm)

        # expansive path
        u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = self.conv2d_block(u6, n_filters=n_filters * 8, kernel_size=kernel, batchnorm=batchnorm)

        u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = self.conv2d_block(u7, n_filters=n_filters * 4, kernel_size=kernel, batchnorm=batchnorm)

        u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = self.conv2d_block(u8, n_filters=n_filters * 2, kernel_size=kernel, batchnorm=batchnorm)

        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        u9 = Dropout(dropout)(u9)
        c9 = self.conv2d_block(u9, n_filters=n_filters * 1, kernel_size=kernel, batchnorm=batchnorm)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
        model = Model(inputs=[input_img], outputs=[outputs])
        return model
