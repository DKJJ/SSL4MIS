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

dir1 = '../../data/hyper/DKJ_pick_split/'
dir2 = '../../data/hyper/data'
with open(dir1 + 'DKJ_pick.txt', 'r') as f1:
    pick_list = f1.readlines()
pick_list = [item.replace('\n', '')
                    for item in pick_list]
list = os.listdir(dir2)
for i in range(len(pick_list)):
    list.remove(pick_list[i])
output = open(dir1 + 'sb.txt', 'w')
for row in list:
    output.write(row)
    output.write('\n')
output.close()