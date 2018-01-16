"""
Some useful functions are defined in this file
Author: Kaihua Tang
"""
from scipy import misc
import scipy.io as scio
import tensorflow as tf
import numpy as np
import time
import math
import random


def random_mini_batches(totalSize, mini_batch_size = 64, random = True):
    """
    totalSize : total num of train image
    mini_batch_size : mini batch size
    return a set of arrays that contains the index from 1 to totalSize, each array is mini_batch_size
    """
    np.random.seed(int(time.time()))        
    m = totalSize                   # number of training examples
    mini_batches = []

    if(random):
        permutation = list(np.random.permutation(m))
    else:
        permutation = list(range(m))

    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batches.append(permutation[k * mini_batch_size : (k + 1) * mini_batch_size])
    
    if m % mini_batch_size != 0:
        mini_batches.append(permutation[(k + 1) * mini_batch_size :])
    
    return mini_batches

def load_all_image(nameList, h, w, c, parentPath, create_npy = False):
    """
    Load all image data in advance
    nameList: name of image we need to load
    """
    all_size = len(nameList)
    all_data = np.zeros((all_size, h, w, c), dtype = "uint8")
    for i in range(all_size):
        tmp_img = load_images(parentPath + nameList[i])
        all_data[i,:,:,:] = tmp_img[:,:,:]
    if(create_npy):
        np.save('./1200_data.npy',all_data)
    return all_data


def get_minibatch(indexList, labelList, h, w, c, n, allImage, is_sparse = False):
    """
    Load one batch images.
    indexList: (size, 1).
    nameList: (totalSize, string).
    labelList: (totalSize, int)
    h, w, c: height, width, channel
    n: number of labels
    """
    m_size = len(indexList)
    batch_X = np.ndarray([m_size, h, w, c])
    if(is_sparse):
        batch_Y = np.zeros((m_size), dtype = 'int64')
    else:
        batch_Y = np.zeros((m_size, n))
    #print(paths)
    for i in range(m_size):
        batch_X[i,:,:,:] = allImage[indexList[i],:,:,:]
        if(is_sparse):
            batch_Y[i] = labelList[indexList[i]] - 1
        else:
            batch_Y[i, :] = [1 if j == (labelList[indexList[i]] - 1) else 0 for j in range(n)]
    return batch_X, batch_Y


def load_images(path):
    """
    Load multiple images.
    :param paths: The image paths.
    """
    img = misc.imread(path, mode="RGB").astype(float)
    return img
