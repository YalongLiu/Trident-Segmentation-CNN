"""
Trident Segmentation CNN
The main model implementation.

Copyright (c) 2017 Yalong Liu
Licensed under the MIT License (see LICENSE for details)
Written by Yalong Liu
"""
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, concatenate, TimeDistributed, \
    Conv2DTranspose, Lambda
import tensorflow as tf

from frames.model import BaseModel


def identity_block(input_tensor, strides):
    x = BatchNormalization()(input_tensor)
    x = Activation(activation='relu')(x)
    x = Conv2D(filters=x.shape[3].value, kernel_size=(3, 3), padding='same', strides=strides[0])(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = Conv2D(filters=x.shape[3].value, kernel_size=(3, 3), padding='same', strides=strides[1])(x)

    ide_path = Add()([x, input_tensor])
    return ide_path


def res_block(x, nb_filters, strides):
    res_path = BatchNormalization()(x)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(res_path)

    shortcut = BatchNormalization()(x)
    shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0])(shortcut)

    res_path = Add()([shortcut, res_path])
    return res_path


def td_res_block(x, nb_filters, strides):
    res_path = TimeDistributed(BatchNormalization())(x)
    res_path = Activation(activation='relu')(res_path)
    res_path = TimeDistributed(Conv2D(nb_filters[0], (3, 3), strides=strides[0], padding='same'))(res_path)
    res_path = TimeDistributed(BatchNormalization())(res_path)
    res_path = Activation(activation='relu')(res_path)
    res_path = TimeDistributed(Conv2D(nb_filters[1], (3, 3), strides=strides[1], padding='same'))(res_path)

    shortcut = TimeDistributed(BatchNormalization())(x)
    shortcut = Activation(activation='relu')(shortcut)
    shortcut = TimeDistributed(Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0]))(shortcut)

    res_path = Add()([shortcut, res_path])
    return res_path


def td_identity_block(inputs, strides):
    x = TimeDistributed(BatchNormalization())(inputs)
    x = Activation(activation='relu')(x)
    x = TimeDistributed(Conv2D(x.shape[-1].value, (3, 3), strides=strides[0], padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = Activation(activation='relu')(x)
    x = TimeDistributed(Conv2D(x.shape[-1].value, (3, 3), strides=strides[1], padding='same'))(x)

    ide_path = Add()([inputs, x])
    return ide_path


def time_distributed_concat(input_tensor):
    c0 = Lambda(lambda xx: xx[:, 0, :, :, :])(input_tensor)
    c1 = Lambda(lambda xx: xx[:, 1, :, :, :])(input_tensor)
    c2 = Lambda(lambda xx: xx[:, 2, :, :, :])(input_tensor)
    x = concatenate([c0, c1, c2], axis=-1)
    return x


def trident_graph(inputs, k):
    x = TimeDistributed(Conv2D(k, (3, 3), padding='same'))(inputs)
    x = TimeDistributed(BatchNormalization())(x)
    x = Activation(activation='relu')(x)
    x = TimeDistributed(Conv2D(k, (3, 3), padding='same'))(x)

    shortcut = TimeDistributed(BatchNormalization())(inputs)
    shortcut = TimeDistributed(Conv2D(k, (1, 1)))(shortcut)

    c1 = Add()([shortcut, x])
    c1c = time_distributed_concat(c1)

    c2 = td_res_block(c1, [2 * k, 2 * k], [(2, 2), (1, 1)])
    c2c = time_distributed_concat(c2)

    c3 = td_res_block(c2, [4 * k, 4 * k], [(2, 2), (1, 1)])
    c3c = time_distributed_concat(c3)

    c4 = td_res_block(c3, [8 * k, 8 * k], [(2, 2), (1, 1)])
    c4c = time_distributed_concat(c4)

    return c1c, c2c, c3c, c4c


class DefineModel(BaseModel):
    def build(self):
        # Build model
        k = self.config.CARDINALITY
        inputs = Input(shape=(3, self.config.PATCH_SIZE[0], self.config.PATCH_SIZE[1], 1), name='inputs',
                       dtype=tf.float32)

        c1c, c2c, c3c, c4c = trident_graph(inputs, k)

        x = Conv2DTranspose(768, (2, 2), strides=2)(c4c)
        x = concatenate([x, c3c], axis=3)
        x = identity_block(x, [(1, 1), (1, 1)])
        x = res_block(x, [256, 256], [(1, 1), (1, 1)])

        x = Conv2DTranspose(256, (2, 2), strides=2)(x)
        x = concatenate([x, c2c], axis=3)
        x = identity_block(x, [(1, 1), (1, 1)])
        x = res_block(x, [128, 128], [(1, 1), (1, 1)])

        x = Conv2DTranspose(128, (2, 2), strides=2)(x)
        x = concatenate([x, c1c], axis=3)
        x = identity_block(x, [(1, 1), (1, 1)])
        x = res_block(x, [64, 64], [(1, 1), (1, 1)])

        outputs = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(x)

        model = Model(inputs=[inputs], outputs=[outputs])
        # model.summary()
        return model
