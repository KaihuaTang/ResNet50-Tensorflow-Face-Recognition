"""
Resnet Test
Get Resnet feature
Author: Kaihua Tang
"""

import math
import time
import tensorflow as tf
import ResNet as resnet
import numpy as np
import scipy.io as scio
from scipy import misc
from utils import *

# image size
WIDTH = 224
HEIGHT = 224
CHANNELS = 3
#Number of output labels
LABELSNUM = 200
#"Learning rate for the gradient descent."
learning_rate = 0.00001
#"Mini batch size"
MINI_BATCH_SIZE = 32
#"num of epochs"
NUM_EPOCHS = 120
#"Path of Label.npy"
label_path = "F:\\OneDrive\\CACDLabel\\label.npy"
#"Path of image file names"
image_name_path = "F:\\OneDrive\\CACDLabel\\name.npy"
# image path
parentPath = "F:\\CACD2000_Crop\\"
# model path. to load
model_path = "./test-save120.npy"
# path of test feature
feature_path = "./resnet_feature.mat"
# test mode
test_mode_on = False
# tensorboard on
tensorboard_on = False
# tensorboard refresh every N batches
tfBoard_refresh = 200
# epoch frequency of saving model
save_frequency = 30


#Lists that store name of image and its label
allNameList = np.load(image_name_path)
allLabelList = np.load(label_path)

"61114 is 801 - 2000"
"162830 is 1991 - 2000"
"150000 is 18XX - 2000"
"14420 is 1 - 200"
testNameList = allNameList[:14420]
testLabelList = allLabelList[:14420]

#num of total training image
num_test_image = testLabelList.shape[0]

res_feature = []

with tf.Session() as sess:

    images = tf.placeholder(tf.float32, shape = [None, WIDTH, HEIGHT, CHANNELS])
    train_mode = tf.placeholder(tf.bool)

    # build resnet model
    resnet_model = resnet.ResNet(ResNet_npy_path = model_path)
    resnet_model.build(images, LABELSNUM, train_mode)

    sess.run(tf.global_variables_initializer())
    resnet_model.set_is_training(False)

    for i in range(14420):
    	if(i%1000 == 0):
    	    	print(i)
    	(minibatch_X, minibatch_Y) = get_minibatch([i], testNameList, testLabelList, HEIGHT, WIDTH, CHANNELS, LABELSNUM)
    	fc1 = sess.run([resnet_model.fc1], feed_dict={images: minibatch_X, train_mode: False})
    	res_feature.append(fc1[0])

    scio.savemat(feature_path,{'feature' : res_feature})
