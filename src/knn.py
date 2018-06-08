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
    
    def __init__(self, sess, class_num, train_data_path="../data/training", test_data_path="../data/testing"):
        self.sess = sess
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.class_num = class_num
        self.training_data = [[], [], [], [], [], [], [], []]
        self.label_set = []
        self.knn_list = []
        #self.pca = PCA(n_components=4096)

        self.inputs = tf.placeholder(tf.float32, [1, 227, 227, 3], name="input_image")
        self.labels = tf.placeholder(tf.float32, [1, 50], name='label')
        self.alexnet = AlexNet(self.inputs, keep_prob=1.0, num_classes=1000, skip_layer=[])
        tf.global_variables_initializer().run()
        self.alexnet.load_initial_weights(sess)

        for _i in range(7):
            nn = KNeighborsClassifier()
            self.knn_list.append(nn)
            #clf = svm.SVC()
            #self.knn_list.append(clf)

        pass

    def load_training_data(self, img_num):
        ds = DataSet(self.train_data_path, 1, self.class_num)
        
            
        for i in range(img_num):
            print(i)
            img, label = ds.next_batch()
            out1, out2, out3, out4, out5, out6, out7 = self.sess.run([self.alexnet.norm1, self.alexnet.norm2, self.alexnet.conv3, self.alexnet.conv4,
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
            
            if i == (img_num - 1):
                self.build_model()
            
            #if (i + 1) % 250 == 0:
                #self.build_model()
            
                

    def load_testing_data(self, img_num, kneighbor):
        ds = DataSet(self.test_data_path, 1, self.class_num)
        source = []
        tar = []
        count = 0
        for i in range(img_num):
            print(i)
            img, label = ds.next_batch()
            #print(img.shape, label.shape)
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
                #print(k)
                tp = [self.label_set[int(j)] for j in self.knn_list[k].kneighbors(source[k], n_neighbors=kneighbor, return_distance=False)[0]]
                tar.extend(tp)
            
            get_target = max(tar, key = tar.count)

            print(target_label, get_target)

            if get_target == target_label:
                count += 1

            if i == img_num - 1:
                print("k: %d, accuracy: %f" % (kneighbor, float(count / img_num)))
            
            '''

            
            print("label: ", np.argmax(label, axis=1)[0])
            ss = []
            ss.append(out1[0].reshape(1, 27 * 27 * 96)[0])
            #print("out1: ", self.knn_list[0].predict(ss))
            #print("out1: ", self.knn_list[0].kneighbors(ss, 5, return_distance=False))
            print("out1: ", [self.label_set[int(i)] for i in self.knn_list[0].kneighbors(ss, 5, return_distance=False)[0]])
            #print("out1: ", self.knn_list[0].predict(ss))
            ss.clear()
            ss.append(out2[0].reshape(1, 13 * 13 * 256)[0])
            #print("out2: ", self.knn_list[1].predict(ss))
            print("out2: ", [self.label_set[int(i)] for i in self.knn_list[1].kneighbors(ss, 5, return_distance=False)[0]])
            #print("out2: ", self.knn_list[1].predict(ss))
            ss.clear()
            ss.append(out3[0].reshape(1, 13 * 13 * 384)[0])
            #print("out3: ", self.knn_list[2].predict(ss))
            print("out3: ", [self.label_set[int(i)] for i in self.knn_list[2].kneighbors(ss, 5, return_distance=False)[0]])
            #print("out3: ", self.knn_list[2].predict(ss))
            ss.clear()
            ss.append(out4[0].reshape(1, 13 * 13 * 384)[0])
            #print("out4: ", self.knn_list[3].predict(ss))
            print("out4: ", [self.label_set[int(i)] for i in self.knn_list[3].kneighbors(ss, 5, return_distance=False)[0]])
            #print("out4: ", self.knn_list[3].predict(ss))
            ss.clear()
            ss.append(out5[0].reshape(1, 6 * 6 * 256)[0])
            #print("out5: ", self.knn_list[4].predict(ss))
            print("out5: ", [self.label_set[int(i)] for i in self.knn_list[4].kneighbors(ss, 5, return_distance=False)[0]])
            #print("out5: ", self.knn_list[4].predict(ss))
            ss.clear()
            ss.append(out6[0])
            #print("out6: ", self.knn_list[5].predict(ss))
            print("out6: ", [self.label_set[int(i)] for i in self.knn_list[5].kneighbors(ss, 5, return_distance=False)[0]])
            #print("out6: ", self.knn_list[5].predict(ss))
            ss.clear()
            ss.append(out7[0])
            #print("out7: ", self.knn_list[6].predict(ss))
            print("out7: ", [self.label_set[int(i)] for i in self.knn_list[6].kneighbors(ss, 5, return_distance=False)[0]])
            #print("out7: ", self.knn_list[6].predict(ss))
            ss.clear()
            '''
                

        pass

    def build_model(self):
        for i in range(7):
            print("build ", i)
            try:
                self.knn_list[i].fit(self.training_data[i + 1], self.label_set)
                self.training_data[i + 1].clear()
                #self.label_set.clear()
            except:
                print("something wrong")

        #self.label_set.clear()
        pass

    

    

if __name__ == "__main__":
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        #knn = KNN(sess, 50, "../data/training/", "../data/test/")
        knn = KNN(sess, 50, "../data/train_augment/", "../data/test_augment/")
        knn.load_training_data(1600)

        for i in range(1, 6):
            knn.load_testing_data(400, i)
    

    '''
    an = AlexNet(img,1,1,[])
    an.create()
    an.load_initial_weights(sess)
    '''


    pass