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
learning_rate_orig = 1e-05
# output Method
final_layer_type ="softmax"
#"Mini batch size"
MINI_BATCH_SIZE = 4
#"num of epochs"
NUM_EPOCHS = 1000
# image path
parentPath = "F:\\CACD2000_Crop\\"
# dictionary path
dicPath = "./newLossDic.npy"
# model path. to load
model_path ="./model/10.npy"
# tensorboard on
tensorboard_on = False
# tensorboard refresh every N batches
tfBoard_refresh = 50
# epoch frequency of saving model
save_frequency = 5
# num of identity
num_identity = 1200
# num of age
num_age = 4
# margin
margin = 0.6

with tf.Session() as sess:

    # build resnet model
    resnet_model = resnet.ResNet(ResNet_npy_path = model_path)

    train_mode = tf.placeholder(tf.bool)
    
    images = tf.placeholder(tf.float32, shape = [None, WIDTH, HEIGHT, CHANNELS])
    resnet_model.build(images, LABELSNUM, train_mode, final_layer_type)
    norm_out = tf.nn.l2_normalize(resnet_model.pool2, -1)

    positive1_out,positive2_out,positive3_out,positive4_out,negative1_out,negative2_out,negative3_out,negative4_out = tf.unstack(tf.reshape(norm_out, [-1,num_age*2,2048]), num_age*2, 1)

    num_minibatches = int(num_identity / MINI_BATCH_SIZE)

    # cost function
    learning_rate = learning_rate_orig
    with tf.name_scope("cost"):
        positive_center = (positive1_out + positive2_out + positive3_out + positive4_out) / num_age

        tripLoss1 = triplet_loss(positive_center, positive1_out, negative1_out, margin)
        tripLoss2 = triplet_loss(positive_center, positive2_out, negative2_out, margin)
        tripLoss3 = triplet_loss(positive_center, positive3_out, negative3_out, margin)
        tripLoss4 = triplet_loss(positive_center, positive4_out, negative4_out, margin)

        costSum = (tripLoss1 + tripLoss2 + tripLoss3 + tripLoss4) / num_age
        cost = tf.reduce_mean(costSum, 0)

    with tf.name_scope("train"):
        global_steps = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(learning_rate_orig, global_steps, num_minibatches * 100, 0.5, staircase = True)
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
        minibatches = random_mini_batches(num_identity, MINI_BATCH_SIZE, random = True)

        # count the number of batch
        batch_index = 0
        for minibatch in minibatches:
            # change learning rate
            #learning_rate = tf.train.exponential_decay(learning_rate_orig, epoch * num_minibatches + batch_index, 50, 0.96, staircase = True)
            sess.run(global_steps.assign(epoch * num_minibatches + batch_index))

            cacl_batch = get_cacl_minibatches(minibatch, num_age, parentPath, num_identity, allNameIndex, WIDTH, HEIGHT, CHANNELS)
            
            # Training and calculating cost
            resnet_model.set_is_training(True)

            temp_cost, _ = sess.run([cost, train], feed_dict = {images : cacl_batch, train_mode: True})
            minibatch_cost += np.sum(temp_cost)
            batch_index += 1

                
            if (batch_index % 50 == 0):
                print("Epoch %i Batch %i Batch Cost %f Learning Rage %f" %(epoch + 1,batch_index, np.sum(temp_cost), sess.run(learning_rate)))


        # print total cost of this epoch
        print("End Epoch %i" % epoch)
        print("Total cost of Epoch %f" % minibatch_cost)

        # save model
        if((epoch + 1) % save_frequency == 0):
            resnet_model.save_npy(sess, './test-save%i.npy' % (epoch + 1))
                







