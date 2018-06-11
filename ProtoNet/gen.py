import sys
sys.path.append('../src')

import numpy as np
import os
import cv2
import random
from utils import *
import argparse
import tensorflow as tf
from alexnet import AlexNet

def flip(img):
    return img[:,::-1,:]

def random_crop(img, factor = 0.95):
    h, w, c = img.shape
    h_out = int(h * factor)
    w_out = int(w * factor)
    h_start = np.random.randint(0, h - h_out)
    w_start = np.random.randint(0, w - w_out)
    output = img[h_start:h_start+h_out, w_start:w_start+w_out]
    return output

def data_augment(data, crop_num):
    aug = []
    for i in range(crop_num):
        temp = cv2.resize(random_crop(data), (227, 227)).astype(np.float32)
        vgg_mean = np.array([103.939, 116.779, 123.68])
        temp =  temp - vgg_mean
        aug.append(temp)
        aug.append(flip(temp))
    return aug

if __name__ == "__main__":
    desc = "Data preparation for Prototypical Network"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--use_valid', type=int, default=1)
    parser.add_argument('--crop_num', type=int, default=4)
    args = parser.parse_args()

    use_valid = (args.use_valid == 1)
    crop_num = args.crop_num
    data_dir = '../data/training'
    test_dir = '../data/testing'
    result_dir = '../data'

    check_folder(data_dir)
    check_folder(test_dir)

    print("[.] Start generating data for ProtoNet...")
    train_data = []
    test_data = []
    train_label = []
    test_label = []
    dir_list = sorted(os.listdir(data_dir))
    for dir_name in dir_list:
        fi_d = sorted(os.listdir(os.path.join(data_dir, dir_name)))
        for file_name in fi_d:
            if file_name.endswith("jpg"):
                data = cv2.imread(data_dir + '/' + dir_name + '/' + file_name)
                label = int(file_name.split('_')[0]) - 1
                print '[Train]', file_name, label
                if use_valid and (file_name.split('_')[1] == '0009.jpg' or file_name.split('_')[1] == '0010.jpg'):
                    aug_data = data_augment(data, 4)
                    test_data.extend(aug_data)
                    test_label.extend([label] * len(aug_data))
                else:
                    aug_data = data_augment(data, 4)
                    train_data.extend(aug_data)
                    train_label.extend([label] * len(aug_data))

    if not use_valid:
        for file_name in os.listdir(test_dir):
            if file_name.endswith("jpg"):
                data = cv2.imread(test_dir + '/' + file_name)
                label = 0
                print '[Test]', file_name
                aug_data = data_augment(data, crop_num)
                test_data.extend(aug_data)
                test_label.extend([label] * len(aug_data))

    train_fc7 = []
    test_fc7 = []
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        inputs = tf.placeholder(tf.float32, [1, 227, 227, 3], name="input_image")
        alexnet = AlexNet(inputs, keep_prob=1.0, num_classes=1000, skip_layer=[])
        tf.global_variables_initializer().run()
        alexnet.load_initial_weights(sess)
        for i in range(len(train_data)):
            fc7 = sess.run([alexnet.fc7], feed_dict={alexnet.X: np.array(train_data[i]).reshape([1, 227, 227, 3])})
            train_fc7.append(fc7[0].reshape([4096]))
        for i in range(len(test_data)):
            fc7 = sess.run([alexnet.fc7], feed_dict={alexnet.X: np.array(test_data[i]).reshape([1, 227, 227, 3])})
            test_fc7.append(fc7[0].reshape([4096]))
    np.save(result_dir + '/train_fc7.npy', np.array(train_fc7))
    np.save(result_dir + '/train_label.npy', np.array(train_label))
    if use_valid:
        np.save(result_dir + '/valid_fc7.npy', np.array(test_fc7))
        np.save(result_dir + '/valid_label.npy', np.array(test_label))
    else:
        np.save(result_dir + '/test_fc7.npy', np.array(test_fc7))
        np.save(result_dir + '/test_label.npy', np.array(test_label)) # No use
    print('[.] Finish!')

            