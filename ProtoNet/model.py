#-*- coding: utf-8 -*-
from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
from ops import *
from utils import *
from dataset import ProtoDataSet

class ProtoNet(object):
    model_name = "ProtoNet"  # name for checkpoint

    def __init__(self, sess, epoch, 
        way, shot, query,
        test_way, test_shot, test_query,
        checkpoint_dir, log_dir, learning_rate = 0.1, beta1=0.5):
        
        self.sess = sess
        self.log_dir = log_dir
        self.epoch = epoch
        self.beta1 = beta1

        self.way = way
        self.shot = shot
        self.query = query
        self.test_way = test_way
        self.test_shot = test_shot
        self.test_query = test_query

        self.train_set = ProtoDataSet('./', 
            self.way, self.query, self.shot, 
            self.test_way, self.test_query, self.test_shot, phase='TRAIN')
        self.log_dir = log_dir + "/train"
        self.checkpoint_dir = checkpoint_dir + "/train"
        # self.test_set = DataSet('../test', self.way, self.query, self.shot, phase='TEST')
        # self.test_log_dir = log_dir + '/test'

        # parameters
        self.input_dim = 4096
        self.inner1 = 4096
        self.output_dim = 4096

        # train
        self.learning_rate = learning_rate

    def encoder(self, x, is_training=True, reuse=False):
        with tf.variable_scope("encoder", reuse=reuse):
            net = tf.nn.relu(bn(linear(x, 8192, scope='fc1'), is_training=is_training, scope='bn1'))
            # net = tf.nn.relu(bn(linear(net, 8192, scope='fc2'), is_training=is_training, scope='bn2'))
            # net = tf.nn.relu(bn(linear(net, 4096, scope='fc3'), is_training=is_training, scope='bn3'))
            out = bn(linear(net, self.output_dim, scope='fc4'), is_training=is_training, scope='bn4')
        return out

    def euclidean_dist(self, x, y):
        N, D = tf.shape(x)[0], tf.shape(x)[1]
        M = tf.shape(y)[0]
        x = tf.tile(tf.expand_dims(x, axis=1), (1, M, 1))
        y = tf.tile(tf.expand_dims(y, axis=0), (N, 1, 1))
        return tf.reduce_mean(tf.square(x - y), axis=2)
    
    def build_model(self):
        """ Graph Input """
        # images
        self.support_set = tf.placeholder(tf.float32, [None, None, self.input_dim], name='support_set')
        self.query_set = tf.placeholder(tf.float32, [None, None, self.input_dim], name='query_set')
        self.label = tf.placeholder(tf.int64, [None, None], name='ground_true_label')
        self.label_one_hot = tf.one_hot(self.label, depth = self.way)

        support_output = self.encoder(tf.reshape(self.support_set, [self.way * self.shot, self.input_dim]))
        query_output = self.encoder(tf.reshape(self.query_set, [self.way * self.query, self.input_dim]), reuse=True)

        output_dim = tf.shape(support_output)[-1]
        c = tf.reduce_mean(tf.reshape(support_output, [self.way, self.shot, output_dim]), axis = 1)
        # (self.way * self.query, self.output_dim) (self.way, self.output_dim)
        # -> (self.way * self.query, self.way)
        dists = self.euclidean_dist(query_output, c)
        log_p = tf.reshape(tf.nn.log_softmax(-dists), [self.way, self.query, -1])
        self.loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(self.label_one_hot, log_p), axis=-1), [-1]))
        self.acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(log_p, axis=-1), self.label)))

        """ Training """
        # divide trainable variables into a group for D and a group for G
        vars = tf.trainable_variables()

        # optimizer
        # self.optim = tf.train.GradientDescentOptimizer(self.learning_rate) \
        #      .minimize(self.loss, var_list=vars)
        self.optim = tf.train.AdamOptimizer(self.learning_rate) \
          .minimize(self.loss, var_list=vars)

        """ Testing """
        test_support_output = self.encoder(tf.reshape(self.support_set, [self.way * self.shot, self.input_dim]), reuse=True)
        test_query_output = self.encoder(tf.reshape(self.query_set, [self.way * self.query, self.input_dim]), reuse=True)
        
        test_output_dim = tf.shape(test_support_output)[-1]
        test_c = tf.reduce_mean(tf.reshape(test_support_output, [self.way, self.shot, output_dim]), axis = 1)
        # (self.way * self.query, self.output_dim) (self.way, self.output_dim)
        # -> (self.way * self.query, self.way)
        test_dists = self.euclidean_dist(test_query_output, test_c)
        test_log_p = tf.reshape(tf.nn.log_softmax(-test_dists), [self.way, self.query, -1])
        self.test_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(self.label_one_hot, test_log_p), axis=-1), [-1]))
        self.test_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(test_log_p, axis=-1), self.label)))

        """ Summary """
        self.loss_sum = tf.summary.scalar("loss", self.loss)
        self.acc_sum = tf.summary.scalar("acc", self.acc)

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        # could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        could_load = False
        if could_load:
            start_epoch = (int)(checkpoint_counter)
            start_batch_id = checkpoint_counter - start_epoch
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            support_set, query_set, labels = self.train_set.next_batch()

            # update network
            _, loss_summary_str, acc_summary_str, loss, acc = \
                self.sess.run([self.optim, self.loss_sum, self.acc_sum, self.loss, self.acc],
                    feed_dict={self.support_set: support_set, self.query_set: query_set, self.label: labels})
            # print one_hot
            self.writer.add_summary(loss_summary_str, counter)
            self.writer.add_summary(acc_summary_str, counter)



            # display training status
            print("[Train Set] Epoch: [%2d] acc: %.8f loss: %.8f" \
                  % (epoch, acc, loss))

            # save model
            self.save(self.checkpoint_dir, counter)

            '''Test'''
            if epoch % 10 == 0:
                acc = 0
                support_set, query_set, labels = self.train_set.get_valid_data()
                test_loss, test_acc = self.sess.run([self.test_loss, self.test_acc], 
                    feed_dict={self.support_set: support_set, self.query_set: query_set, self.label: labels})
                print("[Test Set] Epoch: [%2d] acc: %.8f loss: %.8f" \
                  % (epoch, test_acc, test_loss))
                print("=================================================")



        # save model for final step
        self.save(self.checkpoint_dir, counter)
    
    def pred(self):

        # saver to save model
        self.saver = tf.train.Saver()
        
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        for idx in range(0, self.predict_num_batches):
            support_set, query_set, labels = self.train_set.get_valid_data()
            test_loss, test_acc = self.sess.run([self.test_loss, self.test_acc], 
                feed_dict={self.support_set: support_set, self.query_set: query_set, self.label: labels})
            print test_loss
            print test_acc
            print self.label_name[np.argmax(prob)], self.label_name[np.argmax(labels)]
            print "============" 
    
    @property
    def model_dir(self):
        # return "{}_{}_{}".format(
        #     self.model_name, self.dataset_name,
        #     self.batch_size)
        return ""

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
