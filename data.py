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

    train_dir = '../data/VOC2007/train/'
    train_img_list_path = train_dir + 'ImageSets/Segmentation/train.txt'

    # 训练集数量
    total = num = 20 # 54

    with open(train_img_list_path, 'r') as f:
        img_name_list = f.read().splitlines()

    img_name_list = img_name_list[:num]

    for i, name in enumerate(img_name_list):

        if num <=0:
            break

        # img = cv2.imread(dir + name)
        img = cv2.imread(train_dir + 'JPEGImages/' + name + '.jpg')
        img_resize = cv2.resize(img, (img_size[0], img_size[1]))
        # spe(img_resize.shape, data_set.shape)

        data_set = np.append(data_set, [img_resize.reshape(img_size)], axis=0)
        # spe(data_set.shape)

        # label_img = cv2.imread(label_dir + 'label.png')
        label_img = cv2.imread(train_dir + 'SegmentationClass/' + name + '.png')
        label_img_resize = cv2.resize(label_img, (img_size[0], img_size[1]))

        labels = np.append(labels, [label_img_resize.reshape(img_size)], axis=0)

        num -= 1

    data_set = np.delete(data_set, 0, axis=0)
    labels = np.delete(labels, 0, axis=0)
    # spe(data_set.shape, labels.shape)

    split = 0.8
    train_num = int(total * split)

    X_train, y_train = data_set[:train_num], labels[:train_num]
    X_valid, y_valid = data_set[train_num:], labels[train_num:]
    # spe(X_valid.shape, y_valid.shape)

    return X_train, y_train, X_valid, y_valid

# 准备测试集
def prepare_test_data(img_size=(500, 500, 3)):

    test_set = np.array([np.zeros(img_size)])

    test_dir = '../data/VOC2007/test/'
    test_img_list_path = test_dir + 'ImageSets/Segmentation/test.txt'

    # 训练集数量
    total = num = 10

    with open(test_img_list_path, 'r') as f:
        img_name_list = f.read().splitlines()

    img_name_list = img_name_list[:num]

    for i, name in enumerate(img_name_list):

        if num <=0:
            break

        img = cv2.imread(test_dir + 'JPEGImages/' + name + '.jpg')
        img_resize = cv2.resize(img, (img_size[0], img_size[1]))
        # spe(img_resize.shape, data_set.shape)

        test_set = np.append(test_set, [img_resize.reshape(img_size)], axis=0)
        # spe(data_set.shape)

        num -= 1

    test_set = np.delete(test_set, 0, axis=0)

    return test_set

