#-*- coding: utf-8 -*-

from alexnet import AlexNet
from dataset import DataSet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import LSHForest
from sklearn.decomposition import PCA
from sklearn import svm

class KNN(object):
    
    def __init__(self, sess, class_num, train_data_path="../data/training", test_data_path="../data/testing", kneighbors=10):
        self.sess = sess
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.class_num = class_num
        self.training_data = [[], [], [], [], [], [], [], []]
        self.label_set = []
        self.knn_list = []
        self.kneighbors = kneighbors

        self.inputs = tf.placeholder(tf.float32, [1, 227, 227, 3], name="input_image")
        self.labels = tf.placeholder(tf.float32, [1, 50], name='label')
        self.alexnet = AlexNet(self.inputs, keep_prob=1.0, num_classes=1000, skip_layer=[])
        tf.global_variables_initializer().run()
        self.alexnet.load_initial_weights(sess)

        for _i in range(7):
            nn = KNeighborsClassifier(n_neighbors=kneighbors)
            self.knn_list.append(nn)
        pass

    def load_training_data(self, img_num):
        ds = DataSet(self.train_data_path, 1, self.class_num)
            
        for i in range(img_num):
            img, label = ds.next_batch()
            out1, out2, out3, out4, out5, out6, out7 = self.sess.run([self.alexnet.norm1, self.alexnet.norm2, self.alexnet.conv3, self.alexnet.conv4,
                            self.alexnet.pool5, self.alexnet.fc6, self.alexnet.fc7], feed_dict={self.alexnet.X: img, self.labels: label})
            
            self.training_data[0].append(out1[0].reshape(1, 27 * 27 * 96)[0])
            self.training_data[1].append(out2[0].reshape(1, 13 * 13 * 256)[0])
            self.training_data[2].append(out3[0].reshape(1, 13 * 13 * 384)[0])
            self.training_data[3].append(out4[0].reshape(1, 13 * 13 * 384)[0])
            self.training_data[4].append(out5[0].reshape(1, 6 * 6 * 256)[0])
            self.training_data[5].append(out6[0])
            self.training_data[6].append(out7[0])
            self.label_set.append(np.argmax(label, axis=1)[0])
            
        self.calibration_size = 400
        self.calibration_data = [self.training_data[j][img_num-self.calibration_size:img_num] for j in range(7)]
        self.calibration_label = self.label_set[img_num-self.calibration_size:img_num]
        self.training_data = [self.training_data[j][:img_num-self.calibration_size] for j in range(7)]
        self.label_set = self.label_set[:img_num-self.calibration_size]

        self.build_model()
        self.A = []
        for i in range(self.calibration_size):
            target_label = self.calibration_label[i]
            tar = []
            for k in range(7):
                tp = [self.label_set[int(j)] for j in self.knn_list[k].kneighbors([self.calibration_data[k][i]], n_neighbors=self.kneighbors, return_distance=False)[0]]
                tar.extend(tp)
            total = np.sum(np.not_equal(tar, target_label))
            self.A.append(total)
                
           

    def load_testing_data(self, img_num):
        kneighbor = self.kneighbors
        ds = DataSet(self.test_data_path, 1, self.class_num)
        source = []
        tar = []
        count = 0
        for i in range(img_num):
            print(i)
            img, label = ds.next_batch()
            out1, out2, out3, out4, out5, out6, out7 = self.sess.run([self.alexnet.norm1, self.alexnet.norm2, self.alexnet.conv3, self.alexnet.conv4,
                            self.alexnet.pool5, self.alexnet.fc6, self.alexnet.fc7], feed_dict={self.alexnet.X: img, self.labels: label})

            
            source.clear()
            source.append(out1[0].reshape(1, 27 * 27 * 96))
            source.append(out2[0].reshape(1, 13 * 13 * 256))
            source.append(out3[0].reshape(1, 13 * 13 * 384))
            source.append(out4[0].reshape(1, 13 * 13 * 384))
            source.append(out5[0].reshape(1, 6 * 6 * 256))
            source.append(out6)
            source.append(out7)

            target_label = np.argmax(label, axis=1)[0]

            tar.clear()
            for k in range(7):
                tp = [self.label_set[int(j)] for j in self.knn_list[k].kneighbors(source[k], n_neighbors=kneighbor, return_distance=False)[0]]
                tar.extend(tp)

            max = 0
            m_label = -1
            for k_label in range(self.class_num):
                alpha = np.sum(np.not_equal(tar, k_label))
                pj = np.sum(np.greater_equal(self.A, alpha))
                if pj >= max:
                    max = pj
                    m_label = k_label

            if m_label == target_label:
                count += 1

            if (i + 1) % 20 == 0:
                print("%d / %d, accuracy: %f" % ((i + 1), img_num, float(count / (i + 1))))
          

    def build_model(self):
        for i in range(7):
            print("build ", i)
            try:
                self.knn_list[i].fit(self.training_data[i], self.label_set)
                self.training_data[i].clear()
            except:
                print("something wrong")  

if __name__ == "__main__":
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        knn = KNN(sess, 50, "../data/train_augment/", "../data/test_augment/", 10)
        knn.load_training_data(1600)
        knn.load_testing_data(400)
