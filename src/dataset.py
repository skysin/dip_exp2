import numpy as np
import os
import cv2
import random


class DataSet(object):
    def __init__(self, data_dir, batch_size, label_dim, max_size=-1):
        self.DATA_SIZE = 224
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_size = max_size
        self.gen_image_list()
        self.shuffle_data()
        self.cur_index = 0
        self.end_index = 0
        self.label_dim = label_dim

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
        original_img = cv2.resize(
            img_data, (self.DATA_SIZE, self.DATA_SIZE)) / 255.0
        original_img = original_img * 2 - 1
        return original_img, image_path.split("_")[0][-2:]

    def shuffle_data(self):
        # test set should not be shuffled
        random.shuffle(self.image_list)

    def next_batch(self):
        self.end_index = min(
            [self.cur_index + self.batch_size, self.total_images])
        array_original_img = []
        array_label = []
        for i in range(self.cur_index, self.end_index):
            # print(self.image_list[i])
            original_img, label_img = self.read_image(self.image_list[i])
            # print("label_img: ", label_img)
            array_original_img.append(original_img)
            label = np.zeros([self.label_dim])
            label[int(label_img) - 1] = 1
            array_label.append(label)

        self.cur_index += self.batch_size
        if self.cur_index >= self.total_images:
            self.cur_index = 0
            self.shuffle_data()

        return np.array(array_original_img), np.array(array_label)


if __name__ == "__main__":
    DATA_SET = DataSet("./training", 2, 50, 500)
    for i in range(10):
        original_img, label = DATA_SET.next_batch()
        # cv2.imwrite("hahaha.jpg", original_img[0])
        print("label1:", label[0])
        # print(original_img.shape, label.shape)
