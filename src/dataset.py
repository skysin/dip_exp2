#coding=utf-8   
import numpy as np
import os
import cv2
import random
import fileinput
import tensorflow as tf

from alexnet import AlexNet
from data_augmentation import *
from sklearn.decomposition import PCA

class BaseDataSet(object):
    def __init__(self, data_dir, batch_size, label_dim, data_size=227, max_size=-1):
        self.data_size = data_size
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_size = max_size
        self.gen_image_list()
        self.shuffle_data()
        self.label_dim = label_dim
        self.cur_index = 0
        self.end_index = 0

    def gen_image_list(self):
        self.image_list = []
        self.total_images = 0
        for dir_name in os.listdir(self.data_dir):
            fi_d = os.path.join(self.data_dir, dir_name)
            if not os.path.isdir(os.path.join(self.data_dir, dir_name)):
                continue
            for file_name in os.listdir(fi_d):
                if file_name.endswith("jpg"):
                    if self.max_size > 0 and self.total_images >= self.max_size:
                        break
                    self.image_list.append(dir_name + '/' + file_name)
                    self.total_images += 1
        for file_name in os.listdir(self.data_dir):
            if not os.path.isfile(os.path.join(self.data_dir, file_name)):
                continue
            if file_name.endswith("jpg"):
                if self.max_size > 0 and self.total_images >= self.max_size:
                    break
                self.image_list.append(file_name)
                self.total_images += 1
            
        self.total_images = (self.total_images //
                             self.batch_size) * self.batch_size

        self.total_batches = (self.max_size // self.batch_size) \
            if self.max_size > 0 else (self.total_images // self.batch_size)
        # self.total_batches -= 10

    def read_image(self, image_name):
        image_path = self.data_dir + '/' + image_name
        img_data = cv2.imread(image_path)
        img_data = cv2.resize(img_data, (self.data_size, self.data_size)).astype(np.float32)
        vgg_mean = np.array([103.939, 116.779, 123.68])
        img_data =  img_data - vgg_mean
        #print(image_path, image_path.split("/")[-1].split('_')[0]) 
        return img_data, image_path.split("/")[-1].split('_')[0]

    def shuffle_data(self):
        # test set should not be shuffled
        random.shuffle(self.image_list)

    def next_batch(self):
        pass


class DataSet(BaseDataSet, object):
    def __init__(self, data_dir, batch_size, label_dim, data_size=227, max_size=-1):
        BaseDataSet.__init__(self, data_dir, batch_size,
                             label_dim, data_size, max_size)

    def next_batch(self):
        self.end_index = min(
            [self.cur_index + self.batch_size, self.total_images])
        array_original_img = []
        array_label = []
        for i in range(self.cur_index, self.end_index):
            # print(self.image_list[i])
            original_img, label_img = self.read_image(self.image_list[i])
            #print("label_img: ", label_img)
            array_original_img.append(original_img)
            label = np.zeros([self.label_dim])
            label[int(label_img) - 1] = 1
            array_label.append(label)

        self.cur_index += self.batch_size
        if self.cur_index >= self.total_images:
            self.cur_index = 0
            self.shuffle_data()

        return np.array(array_original_img), np.array(array_label)

# class KNNDataSet(BaseDataSet):
#     def __init__(self, data_dir, batch_size, label_dim, data_size=-1, max_size=-1):
#         super(KNNDataSet,     self).__init__(
#             data_dir, batch_size, label_dim, data_size=-1, max_size=-1)


class ProtoDataSet(BaseDataSet):
    def __init__(self, 
        data_dir, 
        way, query, shot, 
        test_way=None, test_query=None, test_shot=None,
        regen=True, pca=0):

        assert(query == 8)
        assert(test_query == 8)

        self.data_dir = data_dir
        self.way = way
        self.query = query
        self.shot = shot
        self.test_way = test_way
        self.test_query = test_query
        self.test_shot = test_shot

        if regen and pca != 0:
            self.gen_data(pca)

        print("[Dataset] Loading training set...")
        self.correct_map = []
        self.label_set = [[] for i in range(1000)]
        self.classes_list = [i for i in range(1000)]
        self.total_classes = 1000
        self.cur_index = 0
        self.init_correct_map()
        self.init_label_set(pca = (pca != 0))
        self.shuffle_classes_list()

        print("[Dataset] Loading test data...")
        self.test_label_set = [[] for i in range(50)]
        self.test_total_classes = 50
        assert(50 % self.test_way == 0)
        self.test_class_index = 0 - self.test_way
        self.test_cur_index = 0 - self.test_way * self.test_query
        self.load_test_data(pca = (pca != 0))

        print("Finish loading!")

    def gen_data(self, dim):
        print("[Dataset] Regenerate fc7 with compression...")
        print("[Dataset] PCA: dim = " + str(dim))

        fc7 = np.load(self.data_dir + '/fc7.npy')
        train_fc7 = np.load(self.data_dir + '/train_fc7.npy')
        test_fc7 = np.load(self.data_dir + '/valid_fc7.npy')

        data = np.concatenate((fc7, train_fc7, test_fc7), axis=0)
        pca = PCA(n_components=dim, whiten=True)
        result = pca.fit_transform(data)

        fc7 = result[ : fc7.shape[0]]
        train_fc7 = result[fc7.shape[0] : fc7.shape[0] + train_fc7.shape[0]]
        test_fc7 = result[fc7.shape[0] + train_fc7.shape[0] : ]
        np.save(self.data_dir + '/fc7_pca.npy', fc7)
        np.save(self.data_dir + '/train_fc7_pca.npy', train_fc7)
        np.save(self.data_dir + '/valid_fc7_pca.npy', test_fc7)

        print("[Dataset] PCA compression finish!")
        

    def load_test_data(self, pca):
        if pca:
            self.train_fc7 = np.load(self.data_dir + '/train_fc7_pca.npy')
            self.test_fc7 = np.load(self.data_dir + '/valid_fc7_pca.npy')
            self.train_label = np.load(self.data_dir + '/train_label.npy').tolist()
            self.test_label = np.load(self.data_dir + '/valid_label.npy').tolist()
        else:
            self.train_fc7 = np.load(self.data_dir + '/train_fc7.npy')
            self.test_fc7 = np.load(self.data_dir + '/valid_fc7.npy')
            self.train_label = np.load(self.data_dir + '/train_label.npy').tolist()
            self.test_label = np.load(self.data_dir + '/valid_label.npy').tolist()

        for i, val in enumerate(self.train_label):
            self.test_label_set[val].append(i)
        self.test_batch_num = self.test_fc7.shape[0] / (self.test_query * self.test_way)
        assert(self.test_fc7.shape[0] % (self.test_query * self.test_way) == 0)

    def next_test_batch(self):
        self.test_cur_index = (self.test_cur_index + self.test_way * self.test_query) % self.test_fc7.shape[0]

    def repeat_test_batch(self, candidates):
        result_shot = [[] for i in range(self.test_way)]
        for i, candidate in enumerate(candidates):
            goal_set = self.test_label_set[candidate]
            shot_samples = random.sample(range(len(goal_set)), self.test_shot)
            for j in shot_samples:
                result_shot[i].append(self.train_fc7[goal_set[j]])
                # print j, self.train_fc7[goal_set[j]]

        result_query = [[] for i in range(self.test_way)]
        result_label = [[] for i in range(self.test_way)]
        cur_index = self.test_cur_index
        for i in range(self.test_way):
            for j in range(self.test_query):
                result_query[i].append(self.test_fc7[cur_index])
                result_label[i].append(self.test_label[cur_index])
                cur_index = (cur_index + 1) % self.test_fc7.shape[0]
        return np.array(result_shot), np.array(result_query), np.array(result_label)

    def next_batch(self):
        self.end_index = min(
            [self.cur_index + self.way, self.total_classes])
        result_shot = [[] for i in range(self.way)]
        array_label = []
        result_query = [[] for i in range(self.way)]
        total_num = self.shot + self.query
        for i in range(self.cur_index, self.end_index):
            goal_set = self.label_set[self.classes_list[i]]
            temp_total_num = total_num
            if temp_total_num > len(goal_set):
                temp_total_num = len(goal_set)
            array_label.append([i - self.cur_index] * self.query)
            query_samples = []
            shot_samples = random.sample(range(len(goal_set)), temp_total_num)
            for j in range(self.query):
                query_samples.append(shot_samples.pop())
            if len(shot_samples) < self.shot:
                shot_samples.extend(random.sample(shot_samples, self.shot - len(shot_samples)))
            assert(len(shot_samples) == self.shot)
            for j in shot_samples:
                result_shot[i - self.cur_index].append(self.fc7[goal_set[j]])
            for j in query_samples:
                result_query[i - self.cur_index].append(self.fc7[goal_set[j]])
        self.cur_index += self.way
        if self.cur_index >= self.total_classes:
            self.cur_index = 0
            self.shuffle_classes_list()
        # shape(way,shot,4096), shape(way,query,4096), shape(way,)
        return np.array(result_shot), np.array(result_query), np.array(array_label)

    def init_label_set(self, pca):
        labels = np.load(self.data_dir + '/label.npy').tolist()
        if pca:
            self.fc7 = np.load(self.data_dir + '/fc7_pca.npy')
        else:
            self.fc7 = np.load(self.data_dir + '/fc7.npy')
        for i, val in enumerate(labels):
            self.label_set[self.correct_map[val - 1]].append(i)

    def shuffle_classes_list(self):
        random.shuffle(self.classes_list)

    def init_correct_map(self):
        for line in fileinput.input(self.data_dir + "/correct.txt"):
            self.correct_map.append(int(line.split()[1]) - 1)

if __name__ == "__main__":
    DATA_SET = DataSet("../data/train_augment", 2, 50, 227)
    for i in range(10):
        original_img, label = DATA_SET.next_batch()
        # cv2.imwrite("hahaha.jpg", original_img[0])
        print("label1:", label[0])
        # print(original_img.shape, label.shape)
