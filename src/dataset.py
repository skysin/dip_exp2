#coding=utf-8   
import numpy as np
import os
import cv2
import random
import fileinput
import tensorflow as tf

from alexnet import AlexNet

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
        phase="TRAIN", valid=True, gen_test=False):

        print("Data path: " + data_dir)

        self.data_dir = data_dir
        self.way = way
        self.query = query
        self.shot = shot
        self.test_way = test_way
        self.test_query = test_query
        self.test_shot = test_shot
        self.valid = valid
        assert(phase == 'TRAIN' or phase == 'TEST')
        self.phase = phase
        if self.phase == 'TRAIN' and valid:
            assert(test_way != None and test_query != None and test_shot != None)

        if self.phase == 'TRAIN':
            self.correct_map = []
            self.label_set = [[] for i in range(1000)]
            self.classes_list = [i for i in range(1000)]
            self.total_classes = 1000
            self.cur_index = 0
            self.init_correct_map()
            self.init_label_set()
            if valid:
                self.valid_classes_list = self.classes_list[:self.test_way]
                self.classes_list = self.classes_list[self.test_way:]
                print self.valid_classes_list
                self.total_classes -= self.test_way
                self.set_valid_data()
            self.shuffle_classes_list()
        else:
            print("Loading test data...")
            if gen_test or not os.path.exists(data_dir + '/train_fc7.npy') \
                or not os.path.exists(data_dir + '/valid_fc7.npy'):
                self.gen_test_data('/train_augment', '/test_augment')
            self.label_set = [[] for i in range(50)]
            self.classes_list = [i for i in range(50)]
            self.total_classes = 50
            self.cur_index = 0
            self.load_test_data()
            print("Finish loading!")

    def gen_test_data(self, train_dir, test_dir):
        print("Generate fc7 from testing data...")
        train_path = self.data_dir + train_dir
        test_path = self.data_dir + test_dir

        train_data = BaseDataSet(train_path, 500, 50)
        train_data.image_list = sorted(train_data.image_list)
        train_data_set = []
        train_label_set = []
        for image_path in train_data.image_list:
            data, label = train_data.read_image(image_path)
            train_data_set.append(data)
            train_label_set.append(int(label) - 1)

        test_data = BaseDataSet(test_path, 500, 50)
        test_data.image_list = sorted(test_data.image_list)
        test_data_set = []
        test_label_set = []
        for image_path in test_data.image_list:
            data, label = test_data.read_image(image_path)
            test_data_set.append(data)
            test_label_set.append(int(label) - 1)

        train_fc7 = []
        test_fc7 = []
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            inputs = tf.placeholder(tf.float32, [1, 227, 227, 3], name="input_image")
            alexnet = AlexNet(inputs, keep_prob=1.0, num_classes=1000, skip_layer=[])
            tf.global_variables_initializer().run()
            alexnet.load_initial_weights(sess)
            for i in range(len(train_data_set)):
                fc7 = sess.run([alexnet.fc7], feed_dict={alexnet.X: np.array(train_data_set[i]).reshape([1, 227, 227, 3])})
                train_fc7.append(fc7[0].reshape([4096]))
            for i in range(len(test_data_set)):
                fc7 = sess.run([alexnet.fc7], feed_dict={alexnet.X: np.array(test_data_set[i]).reshape([1, 227, 227, 3])})
                test_fc7.append(fc7[0].reshape([4096]))
        np.save(self.data_dir + '/train_fc7.npy', np.array(train_fc7))
        np.save(self.data_dir + '/valid_fc7.npy', np.array(test_fc7))
        np.save(self.data_dir + '/train_label.npy', np.array(train_label_set))
        np.save(self.data_dir + '/valid_label.npy', np.array(test_label_set))
        print('Finish generating!')

    def load_test_data(self):
        train_fc7 = np.load(self.data_dir + '/train_fc7.npy')
        test_fc7 = np.load(self.data_dir + '/valid_fc7.npy')
        train_label = np.load(self.data_dir + '/train_label.npy')
        test_label = np.load(self.data_dir + '/valid_label.npy')
        self.test_support_set = np.zeros([self.way, self.shot, 4096])
        self.test_query_set = np.zeros([self.way, self.query, 4096])
        self.test_label = np.zeros([self.way, self.query])
        for way in range(self.way):
            for shot in range(self.shot):
                self.test_support_set[way, shot, :] = train_fc7[way * 32 + (shot * 4) % 32, :]
        for way in range(self.way):
            for query in range(self.query):
                self.test_query_set[way, query, :] = test_fc7[way * 8 + shot % 8, :]
                self.test_label[way, query] = way

    def next_batch(self):
        if self.phase == 'TEST':
            return self.test_support_set, self.test_query_set, self.test_label

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

    def init_label_set(self):
        labels = np.load(self.data_dir + '/label.npy').tolist()
        self.fc7 = np.load(self.data_dir + '/fc7.npy')
        for i, val in enumerate(labels):
            self.label_set[self.correct_map[val - 1]].append(i)

    def shuffle_classes_list(self):
        random.shuffle(self.classes_list)

    def init_correct_map(self):
        for line in fileinput.input(self.data_dir + "/correct.txt"):
            self.correct_map.append(int(line.split()[1]) - 1)

    def set_valid_data(self):
        result_shot = [[] for i in range(self.test_way)]
        array_label = []
        result_query = [[] for i in range(self.test_way)]
        total_num = self.test_shot + self.test_query
        for i in range(self.test_way):
            goal_set = self.label_set[self.valid_classes_list[i]]
            temp_total_num = total_num
            if temp_total_num > len(goal_set):
                temp_total_num = len(goal_set)
            array_label.append([i] * self.test_query)
            query_samples = []
            shot_samples = random.sample(range(len(goal_set)), temp_total_num)
            for j in range(self.test_query):
                query_samples.append(shot_samples.pop())
            for j in shot_samples:
                result_shot[i].append(self.fc7[goal_set[j]])
            for j in query_samples:
                result_query[i].append(self.fc7[goal_set[j]])
        # shape(way,shot,4096), shape(way,query,4096), shape(way,)
        print np.array(result_shot).shape, np.array(result_query).shape, np.array(array_label).shape
        self.test_support_set = np.array(result_shot)
        self.test_query_set = np.array(result_query)
        self.test_label = np.array(array_label)

    def get_valid_data(self):
        return self.test_support_set, self.test_query_set, self.test_label

if __name__ == "__main__":
    DATA_SET = DataSet("../data/train_augment", 2, 50, 227)
    for i in range(10):
        original_img, label = DATA_SET.next_batch()
        # cv2.imwrite("hahaha.jpg", original_img[0])
        print("label1:", label[0])
        # print(original_img.shape, label.shape)
