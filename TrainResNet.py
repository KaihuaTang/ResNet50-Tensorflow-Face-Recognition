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
learning_rate_orig = 4e-05
# output Method
final_layer_type ="softmax"
#"Mini batch size"
MINI_BATCH_SIZE = 32
#"num of epochs"
NUM_EPOCHS = 1000
#"Path of Label.npy"
label_path = "./CACDLabel/label_1200.npy"
#"Path of image file names"
image_name_path = "./CACDLabel/name_1200.npy"
# image path
parentPath = "F:\\CACD2000_Crop\\"
# model path. to load
model_path = None #"./model/09.npy"
# tensorboard on
tensorboard_on = False
# tensorboard refresh every N batches
tfBoard_refresh = 50
# epoch frequency of saving model
save_frequency = 2
# is sparse softmax
is_sparse = True

#Lists that store name of image and its label
trainNameList = np.load(image_name_path)
trainLabelList = np.load(label_path)

allImageData = load_all_image(trainNameList, HEIGHT, WIDTH, CHANNELS, parentPath)
#allImageData = np.load('./1200_data.npy')

#num of total training image
num_train_image = trainLabelList.shape[0]

with tf.Session() as sess:

    images = tf.placeholder(tf.float32, shape = [None, WIDTH, HEIGHT, CHANNELS])
    if(is_sparse):
        labels = tf.placeholder(tf.int64, shape = [None])
    else:
        labels = tf.placeholder(tf.float32, shape = [None, LABELSNUM])
    train_mode = tf.placeholder(tf.bool)

    # build resnet model
    resnet_model = resnet.ResNet(ResNet_npy_path = model_path)
    resnet_model.build(images, LABELSNUM, train_mode, final_layer_type)

    num_minibatches = int(num_train_image / MINI_BATCH_SIZE)

    # cost function
    learning_rate = learning_rate_orig
    with tf.name_scope("cost"):
        if(final_layer_type == "sigmoid"):
            print("Using weighted sigmoid loss")
            loss = tf.nn.weighted_cross_entropy_with_logits(logits = resnet_model.fc1, targets = labels, pos_weight = 5.0)
        elif(final_layer_type == "softmax" and is_sparse):
            print("Using sparse softmax loss")
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = resnet_model.fc1, labels = labels)
        elif(final_layer_type == "softmax" and (not is_sparse)):
            print("Using softmax loss")
            loss = tf.nn.softmax_cross_entropy_with_logits(logits = resnet_model.fc1, labels = labels)
        cost = tf.reduce_sum(loss)
    with tf.name_scope("train"):
        global_steps = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(learning_rate_orig, global_steps, num_minibatches * 40, 0.5, staircase = True)
        train = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost)

    sess.run(tf.global_variables_initializer())
    print(resnet_model.get_var_count())

    if(tensorboard_on):
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./Tfboard/Result")
        writer.add_graph(sess.graph)

    # used in tensorboard to count record times
    summary_times = 0

    for epoch in range(NUM_EPOCHS):

        print("Start Epoch %i" % (epoch + 1))

        # get index for all mini batches
        minibatch_cost = 0.
        minibatches = random_mini_batches(num_train_image, MINI_BATCH_SIZE, random = True)

        # count the number of batch
        batch_index = 0

        for minibatch in minibatches:
            # get train examples from each mini batch
            (minibatch_X, minibatch_Y) = get_minibatch(minibatch, trainLabelList, HEIGHT, WIDTH, CHANNELS, LABELSNUM, allImageData, is_sparse)

            # change learning rate decay
            sess.run(global_steps.assign(epoch * num_minibatches + batch_index))

            # record examples to monitoring the training process
            if((batch_index % 50 == 0)):
                resnet_model.set_is_training(False)
                prob = sess.run([resnet_model.prob], feed_dict={images: minibatch_X, train_mode: False})
                countMax = np.sum(np.argmax(prob,1) == minibatch_Y)
                print("Epoch %i Batch %i Before Optimization Count %i" %(epoch + 1,batch_index, countMax))
 
            # Training and calculating cost
            resnet_model.set_is_training(True)
            temp_cost, _ = sess.run([cost, train], feed_dict={images: minibatch_X, labels: minibatch_Y, train_mode: True})
            minibatch_cost += np.sum(temp_cost)

            # tensorboard
            if(tensorboard_on) and (batch_index % tfBoard_refresh == 0):
                s = sess.run(merged_summary, feed_dict={images: minibatch_X, labels: minibatch_Y, train_mode: False})
                writer.add_summary(s, summary_times)
                summary_times = summary_times + 1
                # record cost in tensorflow
                tf.summary.scalar('cost', temp_cost)
                #tf.summary.image('input', minibatch_X, 10)
                
            # record examples to monitoring the training process
            if((batch_index % 50 == 0)):
                resnet_model.set_is_training(False)
                prob = sess.run([resnet_model.prob], feed_dict={images: minibatch_X, train_mode: False})
                countMax = np.sum(np.argmax(prob,1) == minibatch_Y)
                print("Epoch %i Batch %i After Optimization Count %i" %(epoch + 1,batch_index, countMax))

            if (batch_index % 50 == 0):
                print("Epoch %i Batch %i Batch Cost %f Learning_rate %f" %(epoch + 1,batch_index, np.sum(temp_cost), sess.run(learning_rate) * 1e10))

            batch_index += 1


        # print total cost of this epoch
        print("End Epoch %i" % (epoch + 1))
        print("Total cost of Epoch %f" % minibatch_cost)

        # save model
        if((epoch + 1) % save_frequency == 0):
            resnet_model.save_npy(sess, './model/model%i.npy' % (epoch + 1))
                







