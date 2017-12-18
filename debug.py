import math
import time
import tensorflow as tf
import ResNet as resnet
import numpy as np
import scipy.io as scio
from scipy import misc
from utils import *

#"Path of Label.npy"
label_path = "F:\\OneDrive\\CACDLabel\\label_train.npy"
#"Path of image file names"
image_name_path = "F:\\OneDrive\\CACDLabel\\name_train.npy"

mat_path = "./Mat.mat"

#Lists that store name of image and its label
allNameList = np.load(image_name_path)
allLabelList = np.load(label_path)

#"61114 is 801 - 2000"
#"162830 is 1991 - 2000"
#"150000 is 18XX - 2000"
#"14419 is 1 - 200"
trainNameList = allNameList[:]
trainLabelList = allLabelList[:]

#num of total training image
num_train_image = trainLabelList.shape[0]

num_minibatches = int(num_train_image / 32)

minibatches = random_mini_batches(num_train_image, 32, random = True)

scio.savemat(mat_path,{'x':minibatches})
