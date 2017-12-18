"""
A Trainable ResNet Class is defined in this file
Author: Kaihua Tang
"""
import math
import numpy as np
import tensorflow as tf
from functools import reduce

VGG_MEAN = [104.7546, 124.328, 167.1754]
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

class ResNet:
	# some properties
    """
    Initialize function
    """
    def __init__(self, ResNet_npy_path=None, trainable=True, open_tensorboard=False, dropout=0.8):
        if ResNet_npy_path is not None:
            self.data_dict = np.load(ResNet_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.open_tensorboard = open_tensorboard
        self.dropout = dropout
        self.is_training = True

    def set_is_training(self, isTrain):
    	self.is_training = isTrain

    def build(self, rgb, label_num, train_mode=None, last_layer_type = "softmax"):
        """
        load variable from npy to build the Resnet or Generate a new one
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        print(bgr.get_shape().as_list())
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1 = self.conv_layer(bgr, 7, 3, 64, 2, "conv1")

        self.pool1 = self.max_pool(self.conv1, 3, 2, "pool1")
        self.block1_1 = self.res_block_3_layers(self.pool1, [64, 64, 256], "block1_1", True)
        self.block1_2 = self.res_block_3_layers(self.block1_1, [64, 64, 256], "block1_2")
        self.block1_3 = self.res_block_3_layers(self.block1_2, [64, 64, 256], "block1_3")

        self.pool2 = self.max_pool(self.block1_3, 2, 2, "pool2")
        self.block2_1 = self.res_block_3_layers(self.pool2, [128, 128, 512], "block2_1", True)
        self.block2_2 = self.res_block_3_layers(self.block2_1, [128, 128, 512], "block2_2")
        self.block2_3 = self.res_block_3_layers(self.block2_2, [128, 128, 512], "block2_3")
        self.block2_4 = self.res_block_3_layers(self.block2_3, [128, 128, 512], "block2_4")

        self.pool3 = self.max_pool(self.block2_4, 2, 2, "pool3")
        self.block3_1 = self.res_block_3_layers(self.pool3, [256, 256, 1024], "block3_1", True)
        self.block3_2 = self.res_block_3_layers(self.block3_1, [256, 256, 1024], "block3_2")
        self.block3_3 = self.res_block_3_layers(self.block3_2, [256, 256, 1024], "block3_3")
        self.block3_4 = self.res_block_3_layers(self.block3_3, [256, 256, 1024], "block3_4")
        self.block3_5 = self.res_block_3_layers(self.block3_4, [256, 256, 1024], "block3_5")
        self.block3_6 = self.res_block_3_layers(self.block3_5, [256, 256, 1024], "block3_6")

        self.pool4 = self.max_pool(self.block3_6, 2, 2, "pool4")
        self.block4_1 = self.res_block_3_layers(self.pool4, [512, 512, 2048], "block4_1", True)
        self.block4_2 = self.res_block_3_layers(self.block4_1, [512, 512, 2048], "block4_2")
        self.block4_3 = self.res_block_3_layers(self.block4_2, [512, 512, 2048], "block4_3")

        self.pool5 = self.avg_pool(self.block4_3, 7, 1, "pool5")
        #self.fc0 = self.fc_layer(self.pool5, 2048, 1024, "fc0") 
        #self.relu1 = tf.nn.relu(self.fc0)
        #if train_mode is not None:
        #    self.relu1 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu1, self.dropout), lambda: self.relu1)
        #elif self.trainable:
        #    self.relu1 = tf.nn.dropout(self.relu1, self.dropout)


        self.fc1 = self.fc_layer(self.pool5, 2048, label_num, "fc1_")

        if(last_layer_type == "sigmoid"):
	        self.prob = tf.nn.sigmoid(self.fc1, name="prob")
        elif(last_layer_type == "softmax"):
        	self.prob = tf.nn.softmax(self.fc1, name="prob")

        self.data_dict = None



    def res_block_3_layers(self, bottom, channel_list, name, change_dimension = False):
    	"""
    	bottom: input values (X)
    	channel_list : number of channel in 3 layers
    	name: block name
    	"""
    	if (change_dimension):
    		block_conv_input = self.conv_layer(bottom, 3, bottom.get_shape().as_list()[-1], channel_list[2], 1, name + "_ShortcutConv")
    	else:
    		block_conv_input = bottom

    	block_conv_1 = self.conv_layer(block_conv_input, 1, channel_list[2], channel_list[0], 1, name + "_lovalConv1")
    	block_norm_1 = tf.layers.batch_normalization(inputs=block_conv_1, axis = 3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.is_training, fused=True)
    	block_relu_1 = tf.nn.relu(block_norm_1)

    	block_conv_2 = self.conv_layer(block_relu_1, 3, channel_list[0], channel_list[1], 1, name + "_lovalConv2")
    	block_norm_2 = tf.layers.batch_normalization(inputs=block_conv_2, axis = 3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.is_training, fused=True)
    	block_relu_2 = tf.nn.relu(block_norm_2)

    	block_conv_3 = self.conv_layer(block_relu_2, 1, channel_list[1], channel_list[2], 1, name + "_lovalConv3")
    	block_res = tf.add(block_conv_input, block_conv_3)
    	relu = tf.nn.relu(block_res)

    	return relu


    def avg_pool(self, bottom, kernal_size = 2, stride = 2, name = "avg"):
    	"""
    	bottom: input values (X)
    	kernal_size : n * n kernal
    	stride : stride
    	name : block_layer name
    	"""
    	print(name + ":")
    	print(bottom.get_shape().as_list())
    	return tf.nn.avg_pool(bottom, ksize=[1, kernal_size, kernal_size, 1], strides=[1, stride, stride, 1], padding='VALID', name=name)

    def max_pool(self, bottom, kernal_size = 2, stride = 2, name = "max"):
    	"""
    	bottom: input values (X)
    	kernal_size : n * n kernal
    	stride : stride
    	name : block_layer name
    	"""
    	print(name + ":")
    	print(bottom.get_shape().as_list())
    	return tf.nn.max_pool(bottom, ksize=[1, kernal_size, kernal_size, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, kernal_size, in_channels, out_channels, stride, name):
    	"""
    	bottom: input values (X)
    	kernal_size : n * n kernal
    	in_channels: number of input filters
    	out_channels : number of output filters
    	stride : stride
    	name : block_layer name
    	"""
    	print(name + ":")
    	print(bottom.get_shape().as_list())
    	with tf.variable_scope(name):
    		filt, conv_biases = self.get_conv_var(kernal_size, in_channels, out_channels, name)

    		conv = tf.nn.conv2d(bottom, filt, [1,stride,stride,1], padding='SAME')
    		bias = tf.nn.bias_add(conv, conv_biases)

    		tf.summary.histogram('weight', filt)
    		tf.summary.histogram('bias', conv_biases)

    		return bias

    def fc_layer(self, bottom, in_size, out_size, name):
    	"""
    	bottom: input values (X)
    	in_size : number of input feature size
    	out_size : number of output feature size
    	"""
    	print(name + ":")
    	print(bottom.get_shape().as_list())
    	with tf.variable_scope(name):
    		weights, biases = self.get_fc_var(in_size, out_size, name)

    		x = tf.reshape(bottom, [-1, in_size])
    		fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

    		tf.summary.histogram('weight', weights)
    		tf.summary.histogram('bias', biases)

    		return fc


    def get_conv_var(self, filter_size, in_channels, out_channels, name):
    	"""
    	filter_size : 3 * 3
    	in_channels : number of input filters
    	out_channels : number of output filters
    	name : block_layer name
    	"""
    	initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, stddev = 1 / math.sqrt(float(filter_size * filter_size)))
    	filters = self.get_var(initial_value, name, 0, name + "_filters")

    	initial_value = tf.truncated_normal([out_channels], 0.0, 1.0)
    	biases = self.get_var(initial_value, name, 1, name + "_biases")

    	return filters, biases

    def get_fc_var(self, in_size, out_size, name):
    	"""
    	in_size : number of input feature size
    	out_size : number of output feature size
    	name : block_layer name
    	"""
    	initial_value = tf.truncated_normal([in_size, out_size], 0.0, stddev = 1 / math.sqrt(float(in_size)))
    	weights = self.get_var(initial_value, name, 0, name + "_weights")

    	initial_value = tf.truncated_normal([out_size], 0.0, 1.0)
    	biases = self.get_var(initial_value, name, 1, name + "_biases")

    	return weights, biases


    def get_var(self, initial_value, name, idx, var_name):
    	"""
    	load variables from Loaded model or new generated random variables
    	initial_value : random initialized value
    	name: block_layer name
    	index: 0,1 weight or bias
    	var_name: name + "_filter"/"_bias"
    	"""
    	if self.data_dict is not None and name in self.data_dict:
    		value = self.data_dict[name][idx]
    	else:
    		value = initial_value

    	if self.trainable:
    		var = tf.Variable(value, name=var_name)
    	else:
    		var = tf.constant(value, dtype=tf.float32, name=var_name)

    	self.var_dict[(name, idx)] = var

    	# print var_name, var.get_shape().as_list()
    	assert var.get_shape() == initial_value.get_shape()

    	return var


    def save_npy(self, sess, npy_path="./Resnet-save.npy"):
    	"""
    	Save this model into a npy file
    	"""
    	assert isinstance(sess, tf.Session)

    	data_dict = {}

    	for (name, idx), var in list(self.var_dict.items()):
    		var_out = sess.run(var)
    		if name not in data_dict:
    			data_dict[name] = {}
    		data_dict[name][idx] = var_out

    	np.save(npy_path, data_dict)
    	print(("file saved", npy_path))
    	return npy_path

    def get_var_count(self):
    	count = 0
    	for v in list(self.var_dict.values()):
    		count += reduce(lambda x, y: x * y, v.get_shape().as_list())
    	return count
