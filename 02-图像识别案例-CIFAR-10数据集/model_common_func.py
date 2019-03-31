"""定义后面都要用到的共享函数"""

import tensorflow as tf

# 定义权值函数
def weight(shape):
    """在构建模型时，需要使用tf.Variable来创建变量，训练时这个变量会不断更新
       这里使用tf.truncated_normal(截断型正态分布)生成标准差为0.1的随机数来初始化权值"""
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='W')

# 定义偏置项，并初始化为0.1
def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape), name='b')

# 定义卷积操作：步长为1，padding为SAME(卷积之后图像的大小不变)
def conv2d(x, W):
    """直接使用tensorflow内置的卷积函数"""
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# 定义池化操作：步长为2，即原尺寸的长宽各除以2
def max_pool_2x2(x):
    """直接使用tensorflow内置的最大值降采样函数"""
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
