import pickle as p
import numpy as np
import os

def load_CIFAR_batch(filename): # 读取一个批次的样本，10000条
    """load single batch of cifar"""
    with open(filename,"rb") as f:
        # 一个样本由标签和图像数据组成
        # <1×label><3072×pixel>(3072=32*32*3)
        data_dict = p.load(f, encoding="bytes")
        images = data_dict[b"data"]
        labels = data_dict[b'labels']

        # 把原始数据结构调整为BCWH(batches,channels,width,height)
        images = images.reshape(10000, 3, 32, 32)   # 四维数组，维度的索引分别为0,1,2,3
        # tensorflow处理数据图像的结构BWHC
        # 所以要把通道数据移动到最后一个维度:transpose函数是按照后面参数(可以理解为围绕的轴)对原数四维组进行转置
        # 对照维度的索引，大致是把索引重新排列以实现转置（0,1,2,3）-（0,2,3,1），就是把原来索引1排到末尾了
        # 一般低维数组（一维二维）用.T进行转置，高维数组用.transpose(索引参数)进行转置
        images = images.transpose(0,2,3,1)
        labels = np.array(labels)
        return images, labels

def load_CIFAR_data(data_dir):   # 完整读取数据集
    """load cifar data"""
    images_train = []
    labels_train = []
    for i in range(5):
        f = os.path.join(data_dir, 'data_batch_%d' % (i+1))
        print('loading', f)
        # 调用load_CIFAR_batch()获得批量的图像及其对应的标签
        image_batch, label_batch = load_CIFAR_batch(f)
        images_train.append(image_batch)
        labels_train.append(label_batch)
        # np.concatenate函数实现将多个数组进行合并成为一个数组，合并结果会降低一维(将索引为0和1的两维合并成一维)
        # 如原先的数组为(4,2,2)，用np.concatenate()后变成两维(8,2)数组
        Xtrain = np.concatenate(images_train)
        Ytrain = np.concatenate(labels_train)
        del image_batch, label_batch

    Xtest, Ytest = load_CIFAR_batch(os.path.join(data_dir, 'test_batch'))
    print('finished loading CIFAR-10 data!')

    # 返回训练集图像和标签，测试集图像和标签
    return Xtrain, Ytrain, Xtest, Ytest


