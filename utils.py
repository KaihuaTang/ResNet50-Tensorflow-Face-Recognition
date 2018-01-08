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

# image path
tripletIndex = "F:\\CACDLabel\\"

def triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
  
    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
    basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return loss

def get_traindata_dictionary(identity, age, path):
	if (path is not None):
		return np.load(path, encoding='latin1').item()
	all_data = {}
	for i in range(identity):
		for j in range(age):
			data_path = tripletIndex + str(i+1) + '_' + str(j+1) + '.mat'
			tmp_data = scio.loadmat(data_path)
			tmp_name = []
			for k in range(tmp_data["name_tmp"].shape[0]):
				tmp_name.append(str(tmp_data["name_tmp"][k][0][0]))
			all_data[(i,j)] = tmp_name
	np.save("./newLossDic.npy",all_data)
	return all_data

def get_newloss_minibatches(num_identity, num_age, allNameIndex, h, w, c, ancher, permutation, age_permutation, parentPath):
	#print("Ancher Identity is: " + str(ancher))
	batch_same = {}
	batch_diff = {}
	for i in range(num_age):
		batch_same[i] = np.ndarray([1, h, w, c])
		batch_diff[i] = np.ndarray([1, h, w, c])

	anti_ancher_list = get_anchor_diff(ancher, num_age, num_identity, permutation)
	for i in range(num_age):
		batch_same[i][0,:,:,:] = load_images(parentPath + random.choice(allNameIndex[(ancher,age_permutation[i])]))
		batch_diff[i][0,:,:,:] = load_images(parentPath + random.choice(allNameIndex[(anti_ancher_list[i],random.randint(0, 9))]))

	return batch_same, batch_diff

		

def get_anchor_diff(ancher, num_age, num_identity, permutation):
	count = 0
	result = []
	while count < num_age:
		t = random.choice(permutation)
		if ((not (t in result)) and (t != ancher)):
			result.append(t)
			count += 1
	return result


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

def load_all_image(nameList, h, w, c, parentPath):
	all_size = len(nameList)
	all_data = np.zeros((all_size, h, w, c), dtype = "uint8")
	for i in range(all_size):
		tmp_img = load_images(parentPath + nameList[i])
		all_data[i,:,:,:] = tmp_img[:,:,:]
	#np.save('./1200_data.npy',all_data)
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
    	# because CACD use 600 - 2000 to train, the following conver them to 0 - 1400
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
	#print(path)
	img = misc.imread(path, mode="RGB").astype(float)
	return img
