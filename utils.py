"""
Some useful functions are defined in this file
Author: Kaihua Tang
"""
from scipy import misc
import numpy as np
import time
import math

# image path
parentPath = "F:\\CACD2000_Crop\\"

def random_mini_batches(totalSize, mini_batch_size = 64, random = True):
	"""
	totalSize : total num of train image
	mini_batch_size : mini batch size

	return a set of arrays that contains the index from 1 to totalSize, each array is mini_batch_size
	"""
    #np.random.seed(1) 
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


def get_minibatch(indexList, nameList, labelList, h, w, c, n):
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
	batch_Y = np.zeros((m_size, n))
	paths = []
	labels = []
	for index in indexList:
		paths.append( parentPath + nameList[index])
		labels.append(labelList[index])
    #print(paths)
	for i in range(m_size):
		batch_X[i,:,:,:] = load_images(paths[i])
    	# because CACD use 600 - 2000 to train, the following conver them to 0 - 1400
		batch_Y[i, :] = [1 if j == (labels[i] - 1) else 0 for j in range(n)]
	return batch_X, batch_Y


def load_images(path):
	"""
	Load multiple images.
	:param paths: The image paths.
	"""
	img = misc.imread(path, mode="RGB").astype(float)
	return img