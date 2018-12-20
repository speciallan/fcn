#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

from fcn.spe import *
import numpy as np
import os

# 准备训练集
def prepare_train_data(img_size=(500, 500, 3)):

    # 构造训练集和验证集
    data_set, labels, test_set = np.array([np.zeros(img_size)]), np.array([np.zeros(img_size)]), np.array([np.zeros(img_size)])

    train_dir = './data/train/'

    train_img_dir = os.listdir(train_dir + 'img/')
    train_img_dir.sort()

    # 训练集数量
    total = num = 20 # 54

    for i, name in enumerate(train_img_dir):

        if num <=0:
            break

        if str(name).lower().endswith('.jpg'):

            # img = cv2.imread(dir + name)
            img = cv2.imread(train_dir + 'img/' + name)
            img_resize = cv2.resize(img, (img_size[0], img_size[1]))
            # spe(img_resize.shape, data_set.shape)

            data_set = np.append(data_set, [img_resize.reshape(img_size)], axis=0)
            # spe(data_set.shape)

            # label_img = cv2.imread(label_dir + 'label.png')
            label_img = cv2.imread(train_dir + 'label/' + name.replace('.jpg', '.png'))
            label_img_resize = cv2.resize(label_img, (img_size[0], img_size[1]))

            labels = np.append(labels, [label_img_resize.reshape(img_size)], axis=0)

            num -= 1

    data_set = np.delete(data_set, 0, axis=0)
    labels = np.delete(labels, 0, axis=0)
    # spe(data_set.shape, labels.shape)

    split = 1
    train_num = int(total * split)

    X_train, y_train = data_set[:train_num], labels[:train_num]
    X_valid, y_valid = data_set[train_num:], labels[train_num:]
    # spe(X_valid.shape, y_valid.shape)

    return X_train, y_train, X_valid, y_valid

# 准备测试集
def prepare_test_data(img_size=(500, 500, 3)):

    test_set = np.array([np.zeros(img_size)])

    test_dir = './data/test/'

    # 训练集数量
    total = num = 20 # 54

    for i, name in enumerate(test_dir):

        if num <=0:
            break

        if str(name).lower().endswith('.jpg'):

            # img = cv2.imread(dir + name)
            img = cv2.imread(test_dir + 'img/' + name)
            img_resize = cv2.resize(img, (img_size[0], img_size[1]))
            # spe(img_resize.shape, data_set.shape)

            test_set = np.append(test_set, [img_resize.reshape(img_size)], axis=0)
            # spe(data_set.shape)

            num -= 1

    test_set = np.delete(test_set, 0, axis=0)

    return test_set

