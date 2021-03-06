#-*- coding: utf-8 -*-
from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
from ops import *
from utils import *
from dataset import DataSet
from alexnet import AlexNet

class transfer_model(object):
    model_name = "transfer_model"     # name for checkpoint

    def __init__(self, sess, epoch, batch_size, checkpoint_dir, log_dir, learning_rate = 0.00001, beta1=0.5):
        self.sess = sess
        self.keep_prob = 1.0
        #self.dataset_name = dataset_name
        #self.result_dir = result_dir
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.beta1 = beta1
        self.label_dim = 50
        self.train_set = DataSet("../data/train_augment", self.batch_size, self.label_dim)
        self.test_set = DataSet("../data/test_augment", self.batch_size, self.label_dim)
		# parameters
        self.input_height = 227
        self.input_width = 227
        #self.output_height = 224
        #self.output_width = 224
        self.c_dim = 3

        # train
        self.init_learning_rate = learning_rate
        
        # get number of batches for a single epoch
        self.num_batches = self.train_set.total_batches
        self.test_num_batches = self.test_set.total_batches

    def classifier(self, x, is_training=True, reuse=False):
        with tf.variable_scope("classifier", reuse=reuse):
            #net = tf.reshape(x, [-1, 6*6*256])
            #net = tf.nn.relu(bn(linear(net, 4096, scope='fc1'), is_training=is_training, scope='bn1'))
            #net = tf.nn.relu(bn(linear(net, 4096, scope='fc2'), is_training=is_training, scope='bn2'))
            #out = linear(net, self.label_dim, scope='fc3')
            out = linear(x, self.label_dim, scope='fc8')
        return out
        
    def build_model(self):
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='input_images')

        # labels
        self.labels = tf.placeholder(tf.float32, [self.batch_size, self.label_dim], name='label')

        # keep prob
        self.keep_prob = tf.placeholder(tf.float32, shape=[])

        # learning rate
        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        """ Loss Function """

        # get fc output of alexnet
        self.alexnet = AlexNet(self.inputs, keep_prob=self.keep_prob, num_classes=1000, skip_layer=[])
        logits = self.classifier(self.alexnet.dropout7, is_training=True, reuse=False)
        prob = tf.nn.softmax(logits)
        self.prob = prob
        # 
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels))

        """ Training """
        # divide trainable variables into a group for D and a group for G
        vars = tf.trainable_variables()

        # optimizer
        # self.optim = tf.train.GradientDescentOptimizer(self.learning_rate) \
        #              .minimize(self.loss, var_list=vars)
        self.optim = tf.train.AdamOptimizer(self.learning_rate) \
                                      .minimize(self.loss, var_list=vars)
        """ Testing """
        # for test
        
        test_logits = self.classifier(self.alexnet.dropout7, is_training=False, reuse=True)
        self.test_prob = tf.nn.softmax(test_logits)
        
        self.correct_pred = tf.equal(tf.argmax(self.test_prob, axis=1), tf.argmax(self.labels, axis=1))
        self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        
        """ Summary """
        self.loss_sum = tf.summary.scalar("loss", self.loss)
        self.acc_sum = tf.summary.scalar("acc", self.acc)

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()
        self.alexnet.load_initial_weights(self.sess)

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        lr = self.init_learning_rate
        min_loss = 100
        update_cnt = 0
        for epoch in range(start_epoch, self.epoch):
            cur_loss = 0
            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                ''' TODO: add data'''
                inputs, labels = self.train_set.next_batch()

                # update network
                _, loss_summary_str, acc_summary_str, loss, prob = self.sess.run([self.optim, self.loss_sum, self.acc_sum, self.loss, self.prob], feed_dict={
                    self.inputs: inputs,
                    self.labels: labels,
                    self.keep_prob: 0.5,
                    self.learning_rate: lr})
                self.writer.add_summary(loss_summary_str, counter)
                self.writer.add_summary(acc_summary_str, counter)
                cur_loss += loss
                #print("output", prob)
                #print("answer", labels)
                # display training status
                counter += 1
                if counter % 100 == 1:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, lr: %.6f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, loss, lr))
            # adapt learning rate
            if cur_loss / (self.num_batches - start_batch_id) > min_loss:
                update_cnt += 1
            else:
                update_cnt = 0
            if update_cnt == 5:
                update_cnt = 0
                lr = lr * 0.5
            
            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            '''Test'''

            
            acc = 0
            for idx in range(0, self.test_num_batches):
                inputs, labels = self.test_set.next_batch()
                correct_pred, prob = self.sess.run([self.correct_pred, self.prob],
                    feed_dict={
                        self.inputs: inputs,
                        self.labels: labels,
                        self.keep_prob: 1.0})
                #if np.sum(prob) == np.argmax(labels):
                #print(prob[0])
                #print(np.argmax(prob, axis=1), np.argmax(labels, axis=1))
                acc += (np.sum(correct_pred) + 0.0)/ self.batch_size
                #print np.sum(correct_pred), self.batch_size
                #print(np.sum(correct_pred), self.batch_size)
                #print(len(correct_pred), labels.shape)
                #print(correct_pred)
            print("Epoch: [%2d] acc: %.8f" \
                      % (epoch, (acc + 0.0) / self.test_num_batches))
            

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
            inputs, labels = self.predict_set.next_batch()
            prob = self.sess.run([self.test_prob], feed_dict={self.inputs: inputs})
            #print prob
            #print labels
            #print self.label_name[np.argmax(prob)], self.label_name[np.argmax(labels)]
            #print "============" 
        
    @property
    def model_dir(self):
        return "{}_{}".format(
            self.model_name,
            self.batch_size)

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
