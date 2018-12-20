#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

from fcn.data import *
from fcn.model import *
from fcn.spe import *

output_dir = base_dir + './data/test/label/'
model_path = base_dir + './fcn.hdf5'

# 准备数据
input_img_size = (500, 500, 3)

X_train, y_train, X_valid, y_valid = prepare_train_data(input_img_size)
X_test = prepare_test_data(input_img_size)

num_classes = 1000

# spe(X_train.shape, y_train.shape, X_valid.shape, X_test.shape)

# 训练模型
model = fcn(input_size=input_img_size, num_classes=num_classes, pre_trained_model=model_path)
model.fit(X_train, y_train, batch_size=1, epochs=1, verbose=1, validation_data=(X_valid, y_valid))

model.summary()

# 预测结果