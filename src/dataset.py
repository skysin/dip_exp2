import numpy as np
import os
import cv2
import random
import fileinput


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

# class KNNDataSet(BaseDataSet):
#     def __init__(self, data_dir, batch_size, label_dim, data_size=-1, max_size=-1):
#         super(KNNDataSet,     self).__init__(
#             data_dir, batch_size, label_dim, data_size=-1, max_size=-1)


class ProtoDataSet(BaseDataSet):
    def __init__(self, data_dir, batch_size, label_dim, data_size=227, max_size=-1):
        BaseDataSet.__init__(self, data_dir, batch_size,
                             label_dim, data_size, max_size)
        self.correct_map = []
        self.label_set = [[] for i in range(1000)]  # 1000个label的样本list
        self.classes_list = [i for i in range(1000)]  # 指向label的key
        self.total_samples = 0  # 总样本数
        self.init_correct_map()
        self.shuffle_classes_list()
        self.init_label_set()
        self.fc7

    def next_batch(self, way, shot, query):
        self.end_index = min(
            [self.cur_index + way, 1000])
        result_shot = [[] for i in range(way)]
        array_label = []
        result_query = [[] for i in range(way)]
        total_num = shot + query
        for i in range(self.cur_index, self.end_index):
            goal_set = self.label_set[self.classes_list[i]]
            array_label.append(self.correct_map[self.classes_list[i]])
            if total_num > len(goal_set):
                total_num = len(goal_set)
            query_samples = []
            shot_samples = random.sample(range(len(goal_set)), total_num)
            for j in range(query):
                query_samples.append(shot_samples.pop())
            for j in shot_samples:
                result_shot[i].append(self.fc7[j])
            for j in query_samples:
                result_query[i].append(self.fc7[j])
        self.cur_index += way
        if self.cur_index >= self.total_samples:
            self.cur_index = 0
            self.shuffle_classes_list()
        # shape(way,shot,4096), shape(way,query,4096), shape(way,)
        return np.array(result_shot), np.array(result_query), np.array(array_label)

    def init_label_set(self):
        labels = np.load('../label.npy').tolist()
        self.fc7 = np.load('../fc7.npy')
        for i, val in enumerate(labels):
            self.label_set[val].append(i)
        self.total_samples = len(labels)

    def shuffle_classes_list(self):
        random.shuffle(self.classes_list)

    def init_correct_map(self):
        for line in fileinput.input("../correct.txt"):
            self.correct_map.append(int(line.split()[1]))



if __name__ == "__main__":
    DATA_SET = DataSet("../data/training", 2, 50, 227)
    for i in range(10):
        original_img, label = DATA_SET.next_batch()
        # cv2.imwrite("hahaha.jpg", original_img[0])
        print("label1:", label[0])
        # print(original_img.shape, label.shape)
