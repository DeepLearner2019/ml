
import numpy as np
import cv2
from sklearn import neighbors
from sklearn.metrics import precision_score
from matplotlib import pyplot as plt
import time
train_dir = 'train'
test_dir = 'test'
def read_image_train(path):
    images = []
    labels = []
    for flower in ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']:
        if flower=='daisy':
           for img_num in range(1, 581, 1):  # 获取指定目录下的所有图片
              img = path + '/' + flower + '/' + "0"+ ' (' + str(img_num)+')' + '.jpg'
              print("reading the image:%s" % img)
              im=cv2.imread(img)
              im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
              #image = cv2.resize(im, (50, 50), interpolation=cv2.INTER_NEAREST)

              images.append(im)
              labels.append([1, 0, 0, 0, 0])
        elif flower=='dandelion':
            for img_num in range(1, 751, 1):  # 获取指定目录下的所有图片
                img = path + '/' + flower + '/' + "1" + ' (' + str(img_num) + ')' + '.jpg'
                print("reading the image:%s" % img)
                im = cv2.imread(img)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                #image = cv2.resize(im, (50, 50), interpolation=cv2.INTER_NEAREST)
                images.append(im)
                labels.append([0, 1, 0, 0, 0])
        elif flower=='roses':
            for img_num in range(1, 591, 1):  # 获取指定目录下的所有图片
                img = path + '/' + flower + '/' + "2" + ' (' + str(img_num) + ')' + '.jpg'
                print("reading the image:%s" % img)
                im = cv2.imread(img)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                #image = cv2.resize(im, (50, 50), interpolation=cv2.INTER_NEAREST)
                images.append(im)
                labels.append([0, 0, 1, 0, 0])
        elif flower=='sunflowers':
            for img_num in range(1, 601, 1):  # 获取指定目录下的所有图片
                img = path + '/' + flower + '/' + "3" + ' (' + str(img_num) + ')' + '.jpg'
                print("reading the image:%s" % img)
                im = cv2.imread(img)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                #image = cv2.resize(im, (50, 50), interpolation=cv2.INTER_NEAREST)
                images.append(im)
                labels.append([0, 0, 0, 1, 0])
        else:
            for img_num in range(1, 691, 1):  # 获取指定目录下的所有图片
                img = path + '/' + flower + '/' + "4" + ' (' + str(img_num) + ')' + '.jpg'
                print("reading the image:%s" % img)
                im = cv2.imread(img)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                #image = cv2.resize(im, (50, 50), interpolation=cv2.INTER_NEAREST)
                images.append(im)
                labels.append([0, 0, 0, 0, 1])
    return images, np.asarray(labels, dtype=np.int32)  # array和asarray都可以将结构数据转化为ndarray，但是主要区别就是当数据源是ndarray时，array仍然会copy出一个副本，占用新的内存，但asarray不会

def read_image_test(path):
    images = []
    labels = []
    for flower in ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']:
        if flower=='daisy':
           for img_num in range(1, 54, 1):  # 获取指定目录下的所有图片
              img = path + '/' + flower + '/' + "0" + ' (' + str(img_num)+')' + '.jpg'
              print("reading the image:%s" % img)
              im = cv2.imread(img)
              im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
              #image = cv2.resize(im, (50, 50), interpolation=cv2.INTER_NEAREST)
              images.append(im)
              labels.append([1, 0, 0, 0, 0])
        elif flower=='dandelion':
            for img_num in range(1, 149, 1):  # 获取指定目录下的所有图片
                img = path + '/' + flower + '/' + "1" + ' (' + str(img_num) + ')' + '.jpg'
                print("reading the image:%s" % img)
                im = cv2.imread(img)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                #image = cv2.resize(im, (50, 50), interpolation=cv2.INTER_NEAREST)
                images.append(im)
                labels.append([0, 1, 0, 0, 0])
        elif flower=='roses':
            for img_num in range(1, 51, 1):  # 获取指定目录下的所有图片
                img = path + '/' + flower + '/' + "2" + ' (' + str(img_num) + ')' + '.jpg'
                print("reading the image:%s" % img)
                im = cv2.imread(img)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                #image = cv2.resize(im, (50, 50), interpolation=cv2.INTER_NEAREST)
                images.append(im)
                labels.append([0, 0, 1, 0, 0])
        elif flower=='sunflowers':
            for img_num in range(1, 91, 1):  # 获取指定目录下的所有图片
                img = path + '/' + flower + '/' + "3" + ' (' + str(img_num) + ')' + '.jpg'
                print("reading the image:%s" % img)
                im = cv2.imread(img)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                #image = cv2.resize(im, (50, 50), interpolation=cv2.INTER_NEAREST)
                images.append(im)
                labels.append([0, 0, 0, 1, 0])
        else:
            for img_num in range(1, 110, 1):  # 获取指定目录下的所有图片
                img = path + '/' + flower + '/' + "4" + ' (' + str(img_num) + ')' + '.jpg'
                print("reading the image:%s" % img)
                im = cv2.imread(img)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                #image = cv2.resize(im, (50, 50), interpolation=cv2.INTER_NEAREST)
                images.append(im)
                labels.append([0, 0, 0, 0, 1])
    return images, np.asarray(labels, dtype=np.int32)  # array和asarray都可以将结构数据转化为ndarray，但是主要区别就是当数据源是ndarray时，array仍然会copy出一个副本，占用新的内存，但asarray不会


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

def texture_detect(train_data,test_data):
    sift = cv2.xfeatures2d.SIFT_create()

    train_features=[]
    for i in range(len(train_data)):

        kp, des = sift.detectAndCompute(train_data[i],None)

        mean = np.mean(des, axis=0)
        var = des.var(axis=0)
        mean = mean[0:45]
        var = var[0:45]
        feature = np.vstack([mean, var])
        feature = np.reshape(feature, (-1, feature.shape[0] * feature.shape[1]))
        feature=np.squeeze(feature)

        train_features.append(feature)
        print(i,'get train-orb:')

    test_features = []
    for i in range(len(test_data)):
        kp, des = sift.detectAndCompute(test_data[i],None)
        mean = np.mean(des, axis=0)
        var = des.var(axis=0)
        mean = mean[0:45]
        var = var[0:45]
        feature = np.vstack([mean, var])
        feature = np.reshape(feature, (-1, feature.shape[0] * feature.shape[1]))
        feature = np.squeeze(feature)

        test_features.append(feature)
        print(i, 'get test-orb:')

    return np.asarray(train_features, dtype=np.float32),np.asarray(test_features, dtype=np.float32)


def getscore(k):
    train_data,train_label=read_image_train(train_dir)
    test_data, test_label =read_image_test(test_dir)
    start = time.clock()
    train_features,test_features = texture_detect(train_data,test_data)
    end = time.clock()
    get_feature_time_test=(end-start)/(450.0/(3210+450.0))
    knn = neighbors.KNeighborsClassifier(k)
    knn.fit(train_features, train_label)
    start = time.clock()
    pred = knn.predict(test_features)
    acc = precision_score(test_label, pred, average='macro')
    elapsed = (time.clock() - start)+get_feature_time_test

    return acc,elapsed

def visaul():
    train_data, train_label = read_image_train(train_dir)
    test_data, test_label = read_image_test(test_dir)

    train_features, test_features = texture_detect(train_data, test_data)

    plt.figure(figsize=(6, 4), dpi=120)
    plt.grid()
    plt.xlabel('The value of K in KNN model')
    plt.ylabel('accuracy')
    dps = []
    accs= []
    for k in range (1,30):

       knn = neighbors.KNeighborsClassifier(k)
       knn.fit(train_features, train_label)
       pred = knn.predict(test_features)
       acc = precision_score(test_label, pred, average='macro')
       dps.append(k)
       accs.append(acc)
    plt.plot(dps, accs, label='test_accuracy')
    plt.legend()
    plt.show()

#visaul()

acc,elapsed=getscore(28)
print("SIFT feature in KNN Model:Accuracy is:",acc)
print("SIFT feature in KNN Model:Time used:", elapsed)