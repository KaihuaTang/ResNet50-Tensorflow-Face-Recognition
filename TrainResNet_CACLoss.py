"""
Used to train ResNet
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
#"Learning rate for the gradient descent."
learning_rate_orig = 1e-04
# output Method
final_layer_type ="softmax"
#"Mini batch size"
MINI_BATCH_SIZE = 1
#"num of epochs"
NUM_EPOCHS = 1000
# image path
parentPath = "F:\\CACD2000_Crop\\"
# dictionary path
dicPath = "./CACDLabel/newLossDic.npy"
# model path. to load
model_path = "./model/10.npy"
# tensorboard on
tensorboard_on = False
# tensorboard refresh every N batches
tfBoard_refresh = 50
# epoch frequency of saving model
save_frequency = 1
# num of identity
num_identity = 1200
# num of age
num_age = 4
# margin
margin = 0.2

with tf.Session() as sess:

    # build resnet model
    resnet_model = resnet.ResNet(ResNet_npy_path = model_path)

    train_mode = tf.placeholder(tf.bool)
    
    positive1 = tf.placeholder(tf.float32, shape = [None, WIDTH, HEIGHT, CHANNELS])
    positive2 = tf.placeholder(tf.float32, shape = [None, WIDTH, HEIGHT, CHANNELS])
    positive3 = tf.placeholder(tf.float32, shape = [None, WIDTH, HEIGHT, CHANNELS])
    positive4 = tf.placeholder(tf.float32, shape = [None, WIDTH, HEIGHT, CHANNELS])

    negative1 = tf.placeholder(tf.float32, shape = [None, WIDTH, HEIGHT, CHANNELS])
    negative2 = tf.placeholder(tf.float32, shape = [None, WIDTH, HEIGHT, CHANNELS])
    negative3 = tf.placeholder(tf.float32, shape = [None, WIDTH, HEIGHT, CHANNELS])
    negative4 = tf.placeholder(tf.float32, shape = [None, WIDTH, HEIGHT, CHANNELS])

    positive1_out = resnet_model.build(positive1, LABELSNUM, train_mode, final_layer_type)
    positive2_out = resnet_model.build(positive2, LABELSNUM, train_mode, final_layer_type)
    positive3_out = resnet_model.build(positive3, LABELSNUM, train_mode, final_layer_type)
    positive4_out = resnet_model.build(positive4, LABELSNUM, train_mode, final_layer_type)
    negative1_out = resnet_model.build(negative1, LABELSNUM, train_mode, final_layer_type)
    negative2_out = resnet_model.build(negative2, LABELSNUM, train_mode, final_layer_type)
    negative3_out = resnet_model.build(negative3, LABELSNUM, train_mode, final_layer_type)
    negative4_out = resnet_model.build(negative4, LABELSNUM, train_mode, final_layer_type)

    positive_pool_1 = tf.nn.l2_normalize(positive1_out,-1)
    positive_pool_2 = tf.nn.l2_normalize(positive2_out,-1)
    positive_pool_3 = tf.nn.l2_normalize(positive3_out,-1)
    positive_pool_4 = tf.nn.l2_normalize(positive4_out,-1)

    negative_pool_1 = tf.nn.l2_normalize(negative1_out,-1)
    negative_pool_2 = tf.nn.l2_normalize(negative2_out,-1)
    negative_pool_3 = tf.nn.l2_normalize(negative3_out,-1)
    negative_pool_4 = tf.nn.l2_normalize(negative4_out,-1)

    print(positive1_out)
    print(positive2_out)
    print(positive3_out)
    print(positive4_out)
    print(negative1_out)
    print(negative2_out)
    print(negative3_out)
    print(negative4_out)

    num_minibatches = int(num_identity / MINI_BATCH_SIZE)

    # cost function
    learning_rate = learning_rate_orig
    with tf.name_scope("cost"):
        positive_center = (positive_pool_1 + positive_pool_2 + positive_pool_3 + positive_pool_4) / num_age

        tripLoss1 = triplet_loss(positive_center, positive_pool_1, negative_pool_1, margin)
        tripLoss2 = triplet_loss(positive_center, positive_pool_2, negative_pool_2, margin)
        tripLoss3 = triplet_loss(positive_center, positive_pool_3, negative_pool_3, margin)
        tripLoss4 = triplet_loss(positive_center, positive_pool_4, negative_pool_4, margin)

        costSum = (tripLoss1 + tripLoss2 + tripLoss3 + tripLoss4) / num_age
        cost = tf.reduce_mean(costSum)

    with tf.name_scope("train"):
        global_steps = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(learning_rate_orig, global_steps, num_minibatches, 0.5, staircase = True)
        train = tf.train.MomentumOptimizer(learning_rate, 0.5).minimize(cost)

    sess.run(tf.global_variables_initializer())
    print(resnet_model.get_var_count())

    if(tensorboard_on):
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./Tfboard/Result")
        writer.add_graph(sess.graph)

    # used in tensorboard to count record times
    summary_times = 0

    allNameIndex = get_traindata_dictionary(num_identity, 10, dicPath)

    for epoch in range(NUM_EPOCHS):

        print("Start Epoch %i" % (epoch + 1))

        # get index for all mini batches
        minibatch_cost = 0.0

        np.random.seed(int(time.time()))        
        m = num_identity                   # number of training examples
        permutation = list(np.random.permutation(m))
        age_permutation = list(np.random.permutation(10))

        # count the number of batch
        batch_index = 0

        for k in range(num_identity):
            # change learning rate
            #learning_rate = tf.train.exponential_decay(learning_rate_orig, epoch * num_minibatches + batch_index, 50, 0.96, staircase = True)
            sess.run(global_steps.assign(epoch * num_minibatches + batch_index))

            batch_same, batch_diff = get_newloss_minibatches(num_identity, num_age, allNameIndex, HEIGHT, WIDTH, CHANNELS, permutation[k], permutation, age_permutation)
 
            # Training and calculating cost
            resnet_model.set_is_training(True)
            feed_dict_tmp = {train_mode: True}
 
            feed_dict_tmp[positive1] = batch_same[0]
            feed_dict_tmp[positive2] = batch_same[1]
            feed_dict_tmp[positive3] = batch_same[2]
            feed_dict_tmp[positive4] = batch_same[3]

            feed_dict_tmp[negative1] = batch_diff[0]
            feed_dict_tmp[negative2] = batch_diff[1]
            feed_dict_tmp[negative3] = batch_diff[2]
            feed_dict_tmp[negative4] = batch_diff[3]

            temp_cost, _ = sess.run([cost, train], feed_dict = feed_dict_tmp)
            minibatch_cost += np.sum(temp_cost)
            batch_index += 1

                
            if (batch_index % 20 == 0):
                print("Epoch %i Batch %i Batch Cost %f Learning Rage %f" %(epoch + 1,batch_index, np.sum(temp_cost), sess.run(learning_rate)))


        # print total cost of this epoch
        print("End Epoch %i" % epoch)
        print("Total cost of Epoch %f" % minibatch_cost)

        # save model
        if((epoch + 1) % save_frequency == 0):
            resnet_model.save_npy(sess, './test-save%i.npy' % (epoch + 1))
                







