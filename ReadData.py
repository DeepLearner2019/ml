import tensorflow as tf
import os
import numpy as np
import cv2
from skimage import io, transform
# 生成训练图片的路径

# 读取图片及其标签函数
def read_image_train(path):
    images = []
    labels = []
    for flower in ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']:
        if flower=='daisy':
           for img_num in range(1, 581, 1):  # 获取指定目录下的所有图片
              img = path + '/' + flower + '/' + "0"+ ' (' + str(img_num)+')' + '.jpg'
              print("reading the image:%s" % img)
              image = io.imread(img)
              image = transform.resize(image, (224, 224, 3))
              images.append(image)
              labels.append([1, 0, 0, 0, 0])
              image = cv2.flip(image, 1, dst=None)  # 水平镜像
              images.append(image)
              labels.append([1, 0, 0, 0, 0])

        elif flower=='dandelion':
            for img_num in range(1, 751, 1):  # 获取指定目录下的所有图片
                img = path + '/' + flower + '/' + "1" + ' (' + str(img_num) + ')' + '.jpg'
                print("reading the image:%s" % img)
                image = io.imread(img)
                image = transform.resize(image, (224, 224, 3))
                images.append(image)
                labels.append([0, 1, 0, 0, 0])
                image = cv2.flip(image, 1, dst=None)  # 水平镜像
                images.append(image)
                labels.append([0, 1, 0, 0, 0])
        elif flower=='roses':
            for img_num in range(1, 591, 1):  # 获取指定目录下的所有图片
                img = path + '/' + flower + '/' + "2" + ' (' + str(img_num) + ')' + '.jpg'
                print("reading the image:%s" % img)
                image = io.imread(img)
                image = transform.resize(image, (224, 224, 3))
                images.append(image)
                labels.append([0, 0, 1, 0, 0])
                image = cv2.flip(image, 1, dst=None)  # 水平镜像
                images.append(image)
                labels.append([0, 0, 1, 0, 0])
        elif flower=='sunflowers':
            for img_num in range(1, 601, 1):  # 获取指定目录下的所有图片
                img = path + '/' + flower + '/' + "3" + ' (' + str(img_num) + ')' + '.jpg'
                print("reading the image:%s" % img)
                image = io.imread(img)
                image = transform.resize(image, (224, 224, 3))
                images.append(image)
                labels.append([0, 0, 0, 1, 0])
                image = cv2.flip(image, 1, dst=None)  # 水平镜像
                images.append(image)
                labels.append([0, 0, 0, 1, 0])
        else:
            for img_num in range(1, 691, 1):  # 获取指定目录下的所有图片
                img = path + '/' + flower + '/' + "4" + ' (' + str(img_num) + ')' + '.jpg'
                print("reading the image:%s" % img)
                image = io.imread(img)
                image = transform.resize(image, (224, 224, 3))
                images.append(image)
                labels.append([0, 0, 0, 0, 1])
                image = cv2.flip(image, 1, dst=None)  # 水平镜像
                images.append(image)
                labels.append([0, 0, 0, 0, 1])
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int32)  # array和asarray都可以将结构数据转化为ndarray，但是主要区别就是当数据源是ndarray时，array仍然会copy出一个副本，占用新的内存，但asarray不会

def read_image_test(path):
    images = []
    labels = []
    for flower in ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']:
        if flower=='daisy':
           for img_num in range(1, 54, 1):  # 获取指定目录下的所有图片
              img = path + '/' + flower + '/' + "0" + ' (' + str(img_num)+')' + '.jpg'
              print("reading the image:%s" % img)
              image = io.imread(img)
              image = transform.resize(image, (224, 224, 3))
              images.append(image)
              labels.append([1, 0, 0, 0, 0])
        elif flower=='dandelion':
            for img_num in range(1, 149, 1):  # 获取指定目录下的所有图片
                img = path + '/' + flower + '/' + "1" + ' (' + str(img_num) + ')' + '.jpg'
                print("reading the image:%s" % img)
                image = io.imread(img)
                image = transform.resize(image, (224, 224, 3))
                images.append(image)
                labels.append([0, 1, 0, 0, 0])
        elif flower=='roses':
            for img_num in range(1, 51, 1):  # 获取指定目录下的所有图片
                img = path + '/' + flower + '/' + "2" + ' (' + str(img_num) + ')' + '.jpg'
                print("reading the image:%s" % img)
                image = io.imread(img)
                image = transform.resize(image, (224, 224, 3))
                images.append(image)
                labels.append([0, 0, 1, 0, 0])
        elif flower=='sunflowers':
            for img_num in range(1, 91, 1):  # 获取指定目录下的所有图片
                img = path + '/' + flower + '/' + "3" + ' (' + str(img_num) + ')' + '.jpg'
                print("reading the image:%s" % img)
                image = io.imread(img)
                image = transform.resize(image, (224, 224, 3))
                images.append(image)
                labels.append([0, 0, 0, 1, 0])
        else:
            for img_num in range(1, 110, 1):  # 获取指定目录下的所有图片
                img = path + '/' + flower + '/' + "4" + ' (' + str(img_num) + ')' + '.jpg'
                print("reading the image:%s" % img)
                image = io.imread(img)
                image = transform.resize(image, (224, 224, 3))
                images.append(image)
                labels.append([0, 0, 0, 0, 1])
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int32)  # array和asarray都可以将结构数据转化为ndarray，但是主要区别就是当数据源是ndarray时，array仍然会copy出一个副本，占用新的内存，但asarray不会


def get_outorder_data_train(path):

    train_data, train_label = read_image_train(path)
    train_image_num = len(train_data)
    train_image_index = np.arange(train_image_num)  # arange(start，stop, step, dtype=None)根据start与stop指定的范围以及step设定的步长，生成一个 ndarray。
    np.random.shuffle(train_image_index)  # 乱序函数，多维时只对一维乱序
    train_data = train_data[train_image_index]  # 乱序后的数据
    train_label = train_label[train_image_index]
    return train_data,train_label

def get_outorder_data_test(path):

    train_data, train_label = read_image_test(path)
    train_image_num = len(train_data)
    train_image_index = np.arange(train_image_num)  # arange(start，stop, step, dtype=None)根据start与stop指定的范围以及step设定的步长，生成一个 ndarray。
    np.random.shuffle(train_image_index)  # 乱序函数，多维时只对一维乱序
    train_data = train_data[train_image_index]  # 乱序后的数据
    train_label = train_label[train_image_index]
    return train_data,train_label

def get_batch(data, label, batch_size):
    for start_index in range(0, len(data) - batch_size + 1, batch_size):
        slice_index = slice(start_index, start_index + batch_size)
        yield data[slice_index], label[slice_index]