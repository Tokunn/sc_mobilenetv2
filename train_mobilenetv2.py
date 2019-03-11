#!/usr/bin/env python3

import sys,os,time
from tqdm import tqdm
#sys.path.append('/home/pi/models/research/slim/')
sys.path.append(os.path.expanduser('~/Documents/tensorflow/mobilenet_v2/models/research/slim/'))
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
import nets.mobilenet_v1
import numpy as np
import input_cifar


cifarpath = "./data/cifar-10-batches-py"
(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS) = (32, 32, 3)

snapshot = sys.argv[1]
ALPHA = float(sys.argv[2])
load_weight = int(sys.argv[3])
div_rate = int(sys.argv[4])
n_batch_size = int(sys.argv[5])
print("snapshotfile:",snapshot,"ALPHA:",ALPHA,"load_weight:",load_weight,"div_rate:",div_rate,"n_batch_size:",n_batch_size)

class ReduceLearningRate(object):
    def __init__(self, init_val, threthold, cnt_max):
        self.counter = 0
        self.learn_rate = init_val
        self.threthold = threthold
        self.cnt_max = cnt_max
    def get_rate(self, val_acc_list):
        if len(val_acc_list) > 1:
            diff = val_acc_list[-1] - val_acc_list[-2]
            #print("val_acu_diff:{:0.4f}".format(diff), flush=True, end=' ')
            if (diff < self.threthold):
                self.counter += 1
                if (self.counter >= self.cnt_max):
                    self.learn_rate /= 10
                    self.counter = 0
                    print("[Reduce learning rate]", flush=True, end=' ')
                #else:
                    #print("                      ", end=' ')
            else:
                self.counter = 0
        print("lean rate : {:0.8f}".format(self.learn_rate), flush=True, end=' ')
        return self.learn_rate

def main():
    x_train, y_train, x_test, y_test, label = input_cifar.get_cifar10(cifarpath)
    n_img = x_train.shape[0] // div_rate
    if (n_img > 0):
        print("n_img:", n_img)
        i = n_img
        choices_list = np.random.choice(x_train.shape[0], i)
        x_train = x_train[choices_list]
        y_train = y_train[choices_list]
    else:
        print("normal")
    N_CLASSES = len(label)
    print(label)


    #############################################################################################


    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope("inputs") as scope:
            input_dims = (None, IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)
            x = tf.placeholder(tf.float32, shape=input_dims, name="X")
            #y = tf.placeholder(tf.int32, shape=[None, N_CLASSES], name="Y")
            y = tf.placeholder(tf.int32, shape=None, name="Y")
            adam_alpha = tf.placeholder_with_default(0.001, shape=None, name="adam_alpha")
            is_training = tf.placeholder_with_default(False, shape=None, name="is_training")
            y_onehot = tf.one_hot(y, depth=N_CLASSES)
            #x_pad = tf.image.resize_image_with_crop_or_pad(x, 160, 160)

        with tf.name_scope("mobilenet_v1") as scope:
            arg_scope = nets.mobilenet_v1.mobilenet_v1_arg_scope()
            with tf.contrib.framework.arg_scope(arg_scope):
                logits, end_points = nets.mobilenet_v1.mobilenet_v1(
                        x,
                        num_classes=N_CLASSES,
                        depth_multiplier=ALPHA,
                        is_training=is_training)

            y_p = tf.nn.softmax(logits, name="output")

        with tf.variable_scope('loss') as scope:
            tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
            #tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
            loss = tf.losses.get_total_loss()

        # Transfer Learning
        with tf.variable_scope('opt') as scope:
            learning_vars = tf.contrib.framework.get_variables('MobilenetV1/Logits')
            optimizer = tf.train.AdamOptimizer(adam_alpha, name="optimizer")
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, var_list=learning_vars, name="train_op")

#        # Fine-Tuning
#        with tf.variable_scope('opt') as scope:
#            optimizer = tf.train.AdamOptimizer(adam_alpha, name="optimizer")
#            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
#            with tf.control_dependencies(update_ops):
#                train_op = optimizer.minimize(loss, name="train_op")

        with tf.variable_scope('eval') as scope:
            accuracy = tf.reduce_mean(
                    tf.cast(
                        tf.equal(tf.argmax(y_p, 1), tf.argmax(y_onehot, 1)), tf.float32)
                    )

        pretrained_include = ["MobilenetV1"]
        pretrained_exclude = ["MobilenetV1/Predictions", "MobilenetV1/Logits"]

        pretrained_vars = tf.contrib.framework.get_variables_to_restore(
                include = pretrained_include,
                exclude = pretrained_exclude)
        pretrained_saver = tf.train.Saver(pretrained_vars, name="pretrained_saver")


        ##########################################################################################


        with tf.Session(graph=graph) as sess:
            with tf.variable_scope('tensorboard') as scope:
                tf.summary_write = tf.summary.FileWriter('logs', graph=sess.graph)
                tf.dummy_summary = tf.summary.scalar(name="dummy", tensor=1)

                #tra_loss_summary = tf.scalar_summary("training_loss", loss)
                #tra_accu_summary = tf.scalar_summary("training_accuracy", accuracy)
                #val_loss_summary = tf.scalar_summary("valication_loss", loss)
                #val_accu_summary = tf.scalar_summary("valication_accuracy", accuracy)

                #writer = tf.train.SummaryWrite("logs", sess.graph_def)

            n_epochs = 200 * div_rate
            print_every = 32
            batch_size = n_batch_size
            steps_per_epoch = len(x_train)//batch_size
            steps_per_epoch_val = len(x_test)//batch_size

            sess.run(tf.global_variables_initializer())
            if load_weight:
                print("Load pretrained weights")
                pretrained_saver.restore(sess, snapshot)
                init_learning_rate = 0.001
            else:
                init_learning_rate = 0.01

            redu_lenrate = ReduceLearningRate(
                    init_val=init_learning_rate, threthold=0.002/div_rate, cnt_max=20*div_rate) # initial value, threthold, cnt
            val_accuracy_list = []

            for epoch in range(n_epochs):
                epoch_start = time.time()
                print("Epoch: {: 3d}/{: 3d} ".format(epoch, n_epochs), end=' ')

                tra_loss = []
                tra_accuracy = []

                learn_rate = redu_lenrate.get_rate(val_accuracy_list)

                #for step in tqdm(range(steps_per_epoch)):
                for step in range(steps_per_epoch):
                    x_batch = x_train[batch_size*step: batch_size*(step+1)]
                    y_batch = y_train[batch_size*step: batch_size*(step+1)]

                    feed_dict = {x : x_batch,
                                 y : y_batch,
                                 adam_alpha : learn_rate,
                                 is_training: True}
                    tmp_loss, _ = sess.run([loss, train_op], feed_dict=feed_dict)
                    tra_loss.append(tmp_loss)

                    feed_dict = {x : x_batch,
                                 y : y_batch,
                                 is_training : False}
                    tmp_acu = sess.run([accuracy], feed_dict=feed_dict)
                    tra_accuracy.append(tmp_acu)

                total_tra_loss = np.average(np.asarray(tra_loss))
                total_tra_acu = np.average(np.asarray(tra_accuracy))

                if (epoch%1==0):
                    val_loss = []
                    val_accuracy = []
                    #for step in tqdm(range(steps_per_epoch_val)):
                    for step in range(steps_per_epoch_val):
                        x_batch = x_test[batch_size*step: batch_size*(step+1)]
                        y_batch = y_test[batch_size*step: batch_size*(step+1)]

                        feed_dict = {x : x_batch,
                                     y : y_batch,
                                     is_training : False}
                        tmp_loss,tmp_acu = sess.run([loss, accuracy], feed_dict=feed_dict)
                        val_loss.append(tmp_loss)
                        val_accuracy.append(tmp_acu)

                    total_val_loss = np.average(np.asarray(val_loss))
                    total_val_acu = np.average(np.asarray(val_accuracy))
                    val_accuracy_list.append(total_val_acu)

                    epoch_finish = time.time()
                    print("time:{:4.4f} tra_loss:{:0.4f} tra_acc:{:0.4f} val_loss:{:0.4f} val_acc:{:0.4f}".format(
                        epoch_finish-epoch_start,
                        total_tra_loss,
                        total_tra_acu,
                        total_val_loss,
                        total_val_acu))

if __name__ == '__main__':
    main()
