#-*- coding: utf-8 -*-
from __future__ import division

import sys
sys.path.append('../src')

import os
import time
import tensorflow as tf
import numpy as np
from ops import *
from utils import *
from dataset import ProtoDataSet
from collections import Counter

class ProtoNet(object):
    model_name = "ProtoNet"  # name for checkpoint

    def __init__(self, sess, epoch, 
        way, shot, query,
        test_way, test_shot, test_query,
        checkpoint_dir, log_dir, learning_rate = 1e-5, beta1=0.5, continue_learn=False):
        
        self.sess = sess
        self.log_dir = log_dir
        self.epoch = epoch
        self.beta1 = beta1
        self.continue_learn = continue_learn

        self.way = way
        self.shot = shot
        self.query = query
        self.test_way = test_way
        self.test_shot = test_shot
        self.test_query = test_query

        self.data = ProtoDataSet('../data', 
            self.way, self.query, self.shot, 
            self.test_way, self.test_query, self.test_shot)
        self.log_dir = log_dir + "/train_6"
        self.checkpoint_dir = checkpoint_dir + "/train_6"

        # parameters
        self.input_dim = 4096
        self.output_dim = 1024

        # train
        print '! learning_rate:', learning_rate
        self.init_learning_rate = learning_rate

    def encoder(self, x, is_training=True, reuse=False):
        with tf.variable_scope("encoder", reuse=reuse):
            linear1 = tf.nn.relu(bn(linear(x, 2048, scope='fc1'), is_training=is_training, scope='bn1'))
            if is_training:
                keep_prob = 0.6
            else:
                keep_prob = 1.0
            drop1 = tf.nn.dropout(linear1, keep_prob=keep_prob)
            # linear2 = tf.nn.relu(bn(linear(linear1, 1024, scope='fc2'), is_training=is_training, scope='bn2'))            
            # drop2 = tf.nn.dropout(linear2, keep_prob=0.6)
            # net3 = tf.nn.relu(bn(linear(net2, 2048, scope='fc3'), is_training=is_training, scope='bn3'))
            # net4 = tf.nn.relu(bn(linear(net3, 1024, scope='fc4'), is_training=is_training, scope='bn4'))
            # net5 = tf.nn.relu(bn(linear(net4, 512, scope='fc5'), is_training=is_training, scope='bn5'))
            out = bn(linear(drop1, self.output_dim, scope='fc6'), is_training=is_training, scope='bn6')
            # tf.summary.histogram('gradient5', tf.gradients(out, net5))
        # self.writer.histogram('bn6', out)
        return out

    def euclidean_dist(self, x, y):
        N, D = tf.shape(x)[0], tf.shape(x)[1]
        M = tf.shape(y)[0]
        x = tf.tile(tf.expand_dims(x, axis=1), (1, M, 1))
        y = tf.tile(tf.expand_dims(y, axis=0), (N, 1, 1))
        return tf.reduce_mean(tf.square(x - y), axis=2)

    def gram_matrix(self, x, y):
        return tf.matmul(x, tf.transpose(y))
    
    def build_model(self):
        """ Graph Input """
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        # images
        self.support_set = tf.placeholder(tf.float32, [None, None, self.input_dim], name='support_set')
        self.query_set = tf.placeholder(tf.float32, [None, None, self.input_dim], name='query_set')
        self.label = tf.placeholder(tf.int64, [None, None], name='ground_true_label')
        self.label_one_hot = tf.one_hot(self.label, depth = self.way)

        support_output = self.encoder(tf.reshape(self.support_set, [self.way * self.shot, self.input_dim]))
        query_output = self.encoder(tf.reshape(self.query_set, [self.way * self.query, self.input_dim]), reuse=True)

        output_dim = tf.shape(support_output)[-1]
        c = tf.reduce_mean(tf.reshape(support_output, [self.way, self.shot, output_dim]), axis = 1)        
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
        self.test_support_output = self.encoder(tf.reshape(self.support_set, [self.way * self.shot, self.input_dim]), reuse=True, is_training=False)
        self.test_query_output = self.encoder(tf.reshape(self.query_set, [self.way * self.query, self.input_dim]), reuse=True, is_training=False)
        
        test_output_dim = tf.shape(self.test_support_output)[-1]
        test_c = tf.reduce_mean(tf.reshape(self.test_support_output, [self.way, self.shot, output_dim]), axis = 1)
        test_dists = self.euclidean_dist(self.test_query_output, test_c)
        self.test_log_p = tf.reshape(tf.nn.log_softmax(-test_dists), [self.way, self.query, -1])
        self.test_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(self.label_one_hot, self.test_log_p), axis=-1), [-1]))
        self.test_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(self.test_log_p, axis=-1), self.label)))

        """ Summary """
        self.loss_sum = tf.summary.scalar("loss", self.loss)
        self.acc_sum = tf.summary.scalar("acc", self.acc)
        self.test_loss_sum = tf.summary.scalar("test_loss", self.test_loss)
        self.test_acc_sum = tf.summary.scalar("test_acc", self.test_acc)

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        if self.continue_learn:
            could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        else:
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
            if self.epoch < 100:
                lr = self.init_learning_rate
            else:
                lr = self.init_learning_rate * 0.1
            support_set, query_set, labels = self.data.next_batch()

            # update network
            _, loss_summary_str, acc_summary_str, loss, acc, summary = \
                self.sess.run([self.optim, self.loss_sum, self.acc_sum, self.loss, self.acc, merged],
                    feed_dict={
                        self.support_set: support_set, 
                        self.query_set: query_set, 
                        self.label: labels, 
                        self.learning_rate: lr})
            self.writer.add_summary(loss_summary_str, epoch)
            self.writer.add_summary(acc_summary_str, epoch)
            self.writer.add_summary(summary, epoch)

            # display training status
            print("[Train Set] Epoch: [%2d] acc: %.8f loss: %.8f" \
                  % (epoch, acc, loss))

            # save model
            self.save(self.checkpoint_dir, counter)
            counter += 1

            '''Test'''
            if epoch % 10 == 0:
                self.test()

        # save model for final step
        self.save(self.checkpoint_dir, self.epoch)

    def get_result(self, results):
        summary = np.zeros([self.test_way, self.test_query, self.test_way])
        for result in results:
            summary += tf.one_hot(tf.argmax(result, axis=-1), depth = self.test_way).eval()
        summary = summary.reshape([self.test_way * self.test_query, -1])
        ans = summary[0::8] + summary[1::8] + summary[2::8] + summary[3::8] + summary[4::8] + summary[5::8] + summary[6::8] + summary[7::8]
        ans = np.argmax(ans, axis=1)
        return ans

    def candidate_next_round(self, batch_result, num):
        counter = Counter(batch_result).most_common(num)
        print counter
        return [item[0] for item in counter]

    def test(self):
        test_fc7 = self.data.test_fc7
        test_label = self.data.test_label
        
        query_set = np.zeros([self.test_way, self.test_query, test_fc7.shape[1]])
        pred = []

        for row in range(test_fc7.shape[0]):
            candidates = range(50)
            query_set = np.tile(test_fc7[row].reshape([1, 1, test_fc7.shape[1]]), (self.test_way, self.test_query, 1))
            
            # First round
            candidate_2 = []
            for i in range(0, len(candidates), self.test_way):
                round_candidate = candidates[i : i + self.test_way]
                batch_result = []
                for repeats in range(5):
                    support_set, _, _ = self.data.repeat_test_batch(round_candidate)
                    test_log_p = self.sess.run([self.test_log_p],
                        feed_dict={self.support_set: support_set, self.query_set: query_set})
                    batch_result.append(np.argmax(test_log_p[0][0, 0, :]))
                candidate_2.extend(round_candidate[self.candidate_next_round(batch_result, 2)])

            # Second round
            candidate_3 = []
            for i in range(0, len(candidate_2), self.test_way):
                round_candidate = candidate_2[i : i + self.test_way]
                batch_result = []
                for repeats in range(5):
                    support_set, _, _ = self.data.repeat_test_batch(round_candidate)
                    test_log_p = self.sess.run([self.test_log_p],
                        feed_dict={self.support_set: support_set, self.query_set: query_set})
                    batch_result.append(np.argmax(test_log_p[0][0, 0, :]))
                candidate_3.extend(round_candidate[self.candidate_next_round(batch_result, 2)])

            # Final round
            round_candidate = candidate_3[0 : self.test_way]
            candidate_4 = candidate_3[self.test_way : ]
            batch_result = []
            for repeats in range(5):
                support_set, _, _ = self.data.repeat_test_batch(round_candidate)
                test_log_p = self.sess.run([self.test_log_p],
                    feed_dict={self.support_set: support_set, self.query_set: query_set})
                batch_result.append(np.argmax(test_log_p[0][0, 0, :]))
            candidate_4.extend(round_candidate[self.candidate_next_round(batch_result, 3)])

            batch_result = []
            for repeats in range(5):
                support_set, _, _ = self.data.repeat_test_batch(candidate_4)
                test_log_p = self.sess.run([self.test_log_p],
                    feed_dict={self.support_set: support_set, self.query_set: query_set})
                batch_result.append(np.argmax(test_log_p[0][0, 0, :]))
            result = round_candidate[self.candidate_next_round(batch_result, 1)][0]
            pred.append(result)

        pred = np.array(pred)
        sum_cnt = test_fc7.shape[0]
        correct_cnt = 0
        for i in range(0, test_fc7.shape[0], 8):
            result = self.candidate_next_round(pred[i : i + 8])[0]
            if result == test_label[i]:
                correct_cnt += 1

        print("[Test Set Summary] Epoch: [%2d] acc: %.8f" \
            % (epoch, 1.0 * correct_cnt / sum_cnt))
        print("=================================================")
    
    def pred(self):

        # saver to save model
        self.saver = tf.train.Saver()
        
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        support_set, query_set, labels = self.test_set.next_batch()
        test_loss, test_acc = self.sess.run([self.test_loss, self.test_acc], 
            feed_dict={self.support_set: support_set, self.query_set: query_set, self.label: labels})
        print test_loss
        print test_acc
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
