#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *
from tensorflow.python.keras.optimizers import *
from fcn.spe import *

def fcn(input_size=(500,500,3), class_num=1000, pre_trained_model=''):

    input = Input(input_size, dtype='float32')

    # 500x500x3
    conv1 = Conv2D(96, 3, padding='same', activation='relu', kernel_initializer='he_normal')(input)
    polling1 = MaxPooling2D()(conv1)

    conv2 = Conv2D(256, 3, padding='same', activation='relu', kernel_initializer='he_normal')(polling1)
    polling2 = MaxPooling2D()(conv2)

    conv3 = Conv2D(384, 3, padding='same', activation='relu', kernel_initializer='he_normal')(polling2)
    conv4 = Conv2D(384, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv3)
    conv5 = Conv2D(256, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv4)

    conv6 = Conv2D(4096, 125, padding='same', activation='relu', kernel_initializer='he_normal')(conv5)
    conv7 = Conv2D(4096, 1, padding='same', activation='relu', kernel_initializer='he_normal')(conv6)
    conv8 = Conv2D(class_num, 1, padding='same', activation='relu', kernel_initializer='he_normal')(conv7)

    conv9 = Conv2DTranspose(3, 500, padding='same', activation='sigmoid', kernel_initializer='he_normal')(conv8)

    # spe(conv9)

    model = Model(input, conv9)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return model
