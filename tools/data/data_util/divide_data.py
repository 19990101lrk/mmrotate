# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import sys

sys.path.append('../../')
import shutil
import os
import random
import math


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

divide_rate = 0.8

# root_path = '/mnt/ExtraDisk/yangxue/data_ship_clean'
root_path = "/root/autodl-tmp/datasets/DOTA"

image_path = root_path + '/ship/split_trainval/images'
ann_path = root_path + '/ship/split_trainval/annfiles'

image_list = os.listdir(image_path)

image_name = [n.split('.')[0] for n in image_list]

random.shuffle(image_name)

train_image = image_name[:int(math.ceil(len(image_name)) * divide_rate)]
test_image = image_name[int(math.ceil(len(image_name)) * divide_rate):]

image_output_trainval = os.path.join(root_path, 'divide_ship/trainval/images')
mkdir(image_output_trainval)
image_output_test = os.path.join(root_path, 'divide_ship/test/images')
mkdir(image_output_test)

ann_trainval = os.path.join(root_path, 'divide_ship/trainval/annfiles')
mkdir(ann_trainval)
ann_test = os.path.join(root_path, 'divide_ship/test/annfiles')
mkdir(ann_test)


count = 0
for i in train_image:
    shutil.copy(os.path.join(image_path, i + '.png'), image_output_trainval)
    shutil.copy(os.path.join(ann_path, i + '.txt'), ann_trainval)
    if count % 1000 == 0:
        print("process step {}".format(count))
    count += 1

for i in test_image:
    shutil.copy(os.path.join(image_path, i + '.png'), image_output_test)
    shutil.copy(os.path.join(ann_path, i + '.txt'), ann_test)
    if count % 1000 == 0:
        print("process step {}".format(count))
    count += 1








