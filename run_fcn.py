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

# spe(X_train.shape, y_train.shape)

# 训练模型
model = fcn()
model.summary()
# model.fit()

# 预测结果