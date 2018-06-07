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

class KNN(object):
    
    def __init__(self, class_num, train_data_path="../data/training"):
        self.train_data_path = train_data_path
        self.class_num = class_num
        self.training_data = [[], [], [], [], [], [], [], []]
        self.label_set = []
        self.knn_list = []
        self.pca = PCA(n_components=500)

        for _i in range(7):
            nn = LSHForest()
            self.knn_list.append(nn)

        pass

    def load_training_data(self, img_num):
        ds = DataSet(self.train_data_path, 1, self.class_num)
        #sess = tf.Session()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            self.inputs = tf.placeholder(tf.float32, [1, 227, 227, 3], name="input_image")
            self.labels = tf.placeholder(tf.float32, [1, 50], name='label')
            self.alexnet = AlexNet(self.inputs, keep_prob=1.0, num_classes=1000, skip_layer=[])
            tf.global_variables_initializer().run()
            self.alexnet.load_initial_weights(sess)
            for i in range(10):
                print(i)
                img, label = ds.next_batch()
                out1, out2, out3, out4, out5, out6, out7 = sess.run([self.alexnet.norm1, self.alexnet.norm2, self.alexnet.conv3, self.alexnet.conv4,
                                self.alexnet.pool5, self.alexnet.fc6, self.alexnet.fc7], feed_dict={self.alexnet.X: img, self.labels: label})
                '''
                print(out1.shape)
                print(out2.shape)
                print(out3.shape)
                print(out4.shape)
                print(out5.shape)
                print(out6.shape)
                print(out7.shape)
                
                (1, 27, 27, 96)
                (1, 13, 13, 256)
                (1, 13, 13, 384)
                (1, 13, 13, 384)
                (1, 6, 6, 256)
                (1, 4096)
                (1, 4096)
                '''
                self.training_data[1].append(out1[0].reshape(1, 27 * 27 * 96)[0])
                self.training_data[2].append(out2[0].reshape(1, 13 * 13 * 256)[0])
                self.training_data[3].append(out3[0].reshape(1, 13 * 13 * 384)[0])
                self.training_data[4].append(out4[0].reshape(1, 13 * 13 * 384)[0])
                self.training_data[5].append(out5[0].reshape(1, 6 * 6 * 256)[0])
                self.training_data[6].append(out6[0])
                self.training_data[7].append(out7[0])
                self.label_set.append(np.argmax(label, axis=1)[0])
                
                if (i + 1) % 10 == 0:
                    self.build_model()

    def load_testing_data(self):
        pass

    def build_model(self):
        for i in range(7):
            print("build ", i)
            newdata = self.pca.fit_transform(self.training_data[i + 1])
            self.knn_list[i].partial_fit(newdata, self.label_set)
            self.training_data[i + 1].clear()
            self.label_set.clear()
        pass

    

    

if __name__ == "__main__":
    knn = KNN(50, "../data/training/")
    knn.load_training_data(500)
    

    '''
    an = AlexNet(img,1,1,[])
    an.create()
    an.load_initial_weights(sess)
    '''


    pass