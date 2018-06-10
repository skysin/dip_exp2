import numpy as np
import os
import cv2
import random
from utils import *

def flip(img):
    return img[:,::-1,:]

def rotate(img):
    h, w, c = img.shape
    output = np.zeros([w, h, c])
    for i in range(h):
        for j in range(w):
            output[j, i] = img[i, j]
    return output

def random_crop(img, factor = 0.75):
    h, w, c = img.shape
    h_out = int(h * factor)
    w_out = int(w * factor)
    h_start = np.random.randint(0, h - h_out)
    w_start = np.random.randint(0, w - w_out)
    output = img[h_start:h_start+h_out, w_start:w_start+w_out]
    return output

if __name__ == "__main__":
    data_dir = '../data/training'
    train_dir = '../data/train_augment'
    test_dir = '../data/test_augment'

    check_folder(train_dir)
    check_folder(test_dir)

    #np.random.seed(1234)

    for dir_name in os.listdir(data_dir):
        fi_d = os.path.join(data_dir, dir_name)
        for file_name in os.listdir(fi_d):
            if file_name.endswith("jpg"):
                if file_name.split('_')[1] == '0009.jpg' or file_name.split('_')[1] == '0010.jpg':
                    output_path = test_dir + '/' + file_name.split('.')[0]
                else:
                    output_path = train_dir + '/' + file_name.split('.')[0]
                
                print(output_path)
                
                image_path = data_dir + '/' + dir_name + '/' + file_name 
                img_data = cv2.imread(image_path)
                cv2.imwrite(output_path + '_origin.jpg', img_data)
                
                flip_img = flip(img_data)
                cv2.imwrite(output_path + '_flip.jpg', flip_img)
                
                rotate_img = rotate(img_data)
                cv2.imwrite(output_path + '_rotate.jpg', rotate_img)
                
                crop_img = random_crop(img_data)
                cv2.imwrite(output_path + '_crop.jpg', crop_img)
            
