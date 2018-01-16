"""
Extract ResNet feature
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
LABELSNUM = 1200
#"Path of Label.npy"
label_path = "./label/label.npy"
#"Path of image file names"
image_name_path = "./label/name.npy"
# image path
parentPath = "F:\\CACD2000_Crop\\"


def CalculateFeature():
    """
    EXtract ResNet Feature by trained model
    model_path: The model we use
    feature_path: The path to save feature
    """
    model_path = "./model/03.npy"
    feature_path = "./resnet_feature.mat"

    #Lists that store name of image and its label
    testNameList = np.load(image_name_path)
    testLabelList = np.load(label_path)

    #num of total training image
    num_test_image = testLabelList.shape[0]
    #load all image data
    allImageData = load_all_image(testNameList, HEIGHT, WIDTH, CHANNELS, parentPath)
    #container for ResNet Feature
    res_feature = np.zeros((num_test_image, 2048))

    with tf.Session() as sess:
        images = tf.placeholder(tf.float32, shape = [None, WIDTH, HEIGHT, CHANNELS])

        # build resnet model
        resnet_model = resnet.ResNet(ResNet_npy_path = model_path)
        resnet_model.build(images, LABELSNUM, "softmax")

        sess.run(tf.global_variables_initializer())
        resnet_model.set_is_training(False)

        for i in range(num_test_image):
           if(i%1000 == 0):
                print(i)
           (minibatch_X, minibatch_Y) = get_minibatch([i], testLabelList, HEIGHT, WIDTH, CHANNELS, LABELSNUM, allImageData, True)
           pool2 = sess.run([resnet_model.pool2], feed_dict={images: minibatch_X})
           res_feature[i][:] = pool2[0][:]

        scio.savemat(feature_path,{'feature' : res_feature})

if __name__ == '__main__':
    CalculateFeature()
