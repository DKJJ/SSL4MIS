import os
# import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
from PIL import Image

seed = 10086
np.random.seed(seed)
folds_number = 5

dir1 = '../../data/hyper/DKJ_pick_split/'
with open(dir1 + 'DKJ_pick.txt', 'r') as f1:
    pick_list = f1.readlines()
pick_list = [item.replace('\n', '')
                    for item in pick_list]
list1 = []
for i in range(20):
    list1.append(pick_list.pop(random.randint(0,len(pick_list)-1)))
output = open(dir1 + 'split_1_train.txt', 'w')
for row in list1:
    output.write(row)
    output.write('\n')
output.close()

list2 = []
for i in range(20):
    list2.append(pick_list.pop(random.randint(0,len(pick_list)-1)))
output = open(dir1 + 'split_1_val.txt', 'w')
for row in list2:
    output.write(row)
    output.write('\n')
output.close()

output = open(dir1 + 'split_1_test.txt', 'w')
for row in pick_list:
    output.write(row)
    output.write('\n')
output.close()