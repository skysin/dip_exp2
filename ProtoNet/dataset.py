import numpy as np
import os
import cv2
import random
import fileinput
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
            for file_name in os.listdir(fi_d):
                if file_name.endswith("jpg"):
                    # file_size = os.path.getsize(self.data_dir + "/" + file_name)
                    # if file_size / 1024 < 5:
                    #     continue
                    self.image_list.append(dir_name + '/' + file_name)
                    self.total_images += 1
                    if (self.max_size > 0 and self.total_images >= self.max_size):
                        break
        self.total_images = (self.total_images //
                             self.batch_size) * self.batch_size

        self.total_batches = (self.max_size // self.batch_size) \
            if self.max_size > 0 else (self.total_images // self.batch_size)
        # self.total_batches -= 10

    def read_image(self, image_name):
        image_path = os.path.join(self.data_dir, image_name)
        img_data = cv2.imread(image_path)
        # img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(image_path, img_data)
        # img_data = cv2.imread(image_path)
        # height, weight, channels = img_data.shape
        #original_img = cv2.resize(img_data, (self.DATA_SIZE, self.DATA_SIZE))
        img_data = cv2.resize(img_data, (self.data_size, self.data_size)).astype(np.float32)
        vgg_mean = np.array([103.939, 116.779, 123.68])
        img_data =  img_data - vgg_mean
        return img_data, image_path.split("_")[0][-2:]

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

class ProtoDataSet(BaseDataSet):
    def __init__(self, 
        data_dir, 
        way, query, shot, 
        test_way=None, test_query=None, test_shot=None,
        phase="TRAIN", valid=True):
        
        # BaseDataSet.__init__(self, data_dir, batch_size,
        #                      label_dim, data_size, max_size)

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

        self.correct_map = []
        self.label_set = [[] for i in range(1000)]
        self.classes_list = [i for i in range(1000)]
        self.total_classes = 1000
        self.cur_index = 0
        self.init_correct_map()
        # self.shuffle_classes_list()
        self.init_label_set()

        if self.phase == 'TRAIN' and valid:
            self.valid_classes_list = self.classes_list[:self.test_way]
            self.classes_list[self.test_way:]
            self.total_classes -= self.test_way
            self.set_valid_data()

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
        # pca_dim = 256

        labels = np.load('./label.npy').tolist()
        self.fc7 = np.load('./fc7.npy')
        # pca = PCA(n_components=pca_dim)
        # self.fc7 = pca.fit_transform(temp_fc7)
        for i, val in enumerate(labels):
            # print '!', val
            self.label_set[self.correct_map[val - 1]].append(i)
        # self.total_samples = len(labels)

    def shuffle_classes_list(self):
        random.shuffle(self.classes_list)

    def init_correct_map(self):
        for line in fileinput.input("correct.txt"):
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
    # DATA_SET = DataSet("../data/training", 2, 50, 227)
    # for i in range(10):
    #     original_img, label = DATA_SET.next_batch()
    #     # cv2.imwrite("hahaha.jpg", original_img[0])
    #     print("label1:", label[0])
    #     # print(original_img.shape, label.shape)

    np.set_printoptions(precision=5, edgeitems=50)
    dataset = ProtoDataSet('./', 50, 10, 10, 50, 10, 10, phase='TRAIN')
    support, query, label = dataset.next_batch()
    print np.mean(support.reshape([50, 10 * 4096]), axis=1).shape
    print np.mean(support.reshape([50, 10 * 4096]), axis=1)
    support = support.reshape([500, 4096])
    W = np.ones([support.shape[1], 256])
    mul = np.matmul(support, W)