"""
Used to train ResNet-50
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
#"Mini batch size"
MINI_BATCH_SIZE = 32
#"Path of Label.npy"
label_path = "./label/label_1200.npy"
#"Path of image file names"
image_name_path = "./label/name_1200.npy"
# image path
parentPath = "F:\\CACD2000_Crop\\"
# data Path: n * 224 * 224 * 3 numpy matrix
data_path = 'F:\\Dataset\\1200_data.npy'



def Train():
    """
    HyperParameters of the Net
    model_path: path of pretrained model, set None if there is no such a model.
    LABELSNUM: Number of output labels
    learning_rate_orig : original learning rate
    NUM_EPOCHS: number of epochs
    save_frequency: frequency of saving model (number of epoches)
    """
    model_path ="./model/03.npy"
    LABELSNUM = 1200
    learning_rate_orig = 1e-05
    NUM_EPOCHS = 1000
    save_frequency = 2
    """
    Classification Layer
    final_layer_type: softmax or sigmoid
    is_sparse: when final layer is softmax, is it sparse
    """
    final_layer_type ="softmax"
    is_sparse = True
    """
    Tensorboard Setting
    tensorboard_on: Turn on Tensorboard or not
    TensorBoard_refresh: refresh rate (number of batches)
    monitoring_rate: Print output rate
    """
    tensorboard_on = False
    TensorBoard_refresh = 50
    monitoring_rate = 50
    
    #Lists that store name of image and its label
    trainNameList = np.load(image_name_path)
    trainLabelList = np.load(label_path)
    if(data_path is None):
        allImageData = load_all_image(trainNameList, HEIGHT, WIDTH, CHANNELS, parentPath)
    else:
        allImageData = np.load(data_path)

    #num of total training image
    num_train_image = trainLabelList.shape[0]

    with tf.Session() as sess:
        images = tf.placeholder(tf.float32, shape = [None, WIDTH, HEIGHT, CHANNELS])
        if(is_sparse):
            labels = tf.placeholder(tf.int64, shape = [None])
        else:
            labels = tf.placeholder(tf.float32, shape = [None, LABELSNUM])

        # build resnet model
        resnet_model = resnet.ResNet(ResNet_npy_path = model_path)
        resnet_model.build(images, LABELSNUM, final_layer_type)
        # number of batches per epoch
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
            learning_rate = tf.train.exponential_decay(learning_rate_orig, global_steps, num_minibatches * 40, 0.1, staircase = True)
            #train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
            #train = tf.train.AdamOptimizer(learning_rate).minimize(cost)
            train = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost)

        sess.run(tf.global_variables_initializer())
        print(resnet_model.get_var_count())

        if(tensorboard_on):
            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter("./TensorBoard/Result")
            writer.add_graph(sess.graph)
            # used in tensorboard to count record times
            summary_times = 0

        for epoch in range(NUM_EPOCHS):
            print("Start Epoch %i" % (epoch + 1))

            minibatch_cost = 0.0
            # count the number of batch
            batch_index = 0  
            # get index for all mini batches
            minibatches = random_mini_batches(num_train_image, MINI_BATCH_SIZE, random = True) 

            for minibatch in minibatches:
                # get train examples from each mini batch
                (minibatch_X, minibatch_Y) = get_minibatch(minibatch, trainLabelList, HEIGHT, WIDTH, CHANNELS, LABELSNUM, allImageData, is_sparse)

                # change learning rate
                sess.run(global_steps.assign(epoch * num_minibatches + batch_index))

                # record examples to monitoring the training process
                if((batch_index % monitoring_rate == 0)):
                    resnet_model.set_is_training(False)
                    fc1, prob = sess.run([resnet_model.fc1, resnet_model.prob], feed_dict={images: minibatch_X})
                    countMax = np.sum(np.argmax(prob,1) == minibatch_Y)
                    print("Epoch %i Batch %i Before Optimization Count %i" %(epoch + 1,batch_index, countMax))
 
                # Training and calculating cost
                resnet_model.set_is_training(True)
                temp_cost, _ = sess.run([cost, train], feed_dict={images: minibatch_X, labels: minibatch_Y})
                minibatch_cost += np.sum(temp_cost)

                # tensorboard
                if(tensorboard_on) and (batch_index % tfBoard_refresh == 0):
                    s = sess.run(merged_summary, feed_dict={images: minibatch_X, labels: minibatch_Y})
                    writer.add_summary(s, summary_times)
                    summary_times = summary_times + 1
                    # record cost in tensorflow
                    tf.summary.scalar('cost', temp_cost)
                
                # record examples to monitoring the training process
                if((batch_index % monitoring_rate == 0)):
                    resnet_model.set_is_training(False)
                    fc1, prob = sess.run([resnet_model.fc1, resnet_model.prob], feed_dict={images: minibatch_X})
                    countMax = np.sum(np.argmax(prob,1) == minibatch_Y)
                    print("Epoch %i Batch %i After Optimization Count %i" %(epoch + 1,batch_index, countMax))
                    # Temp Cost & learning rate
                    print("Epoch %i Batch %i Batch Cost %f Learning_rate %f" %(epoch + 1,batch_index, np.sum(temp_cost), sess.run(learning_rate) * 1e10))

                batch_index += 1


            # print total cost of this epoch
            print("End Epoch %i" % (epoch + 1))
            print("Total cost of Epoch %f" % minibatch_cost)

            # save model
            if((epoch + 1) % save_frequency == 0):
                resnet_model.save_npy(sess, './model/temp-model%i.npy' % (epoch + 1))

if __name__ == '__main__':
    Train()
