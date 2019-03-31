import tensorflow as tf
import numpy as np

# tf.nn.conv2d(input,filter,strides,padding,use_cudnn_on_gpu=None,name=None)
# 定义inputs和卷积核
input_data = tf.Variable(np.random.rand(10,9,9,4),dtype=np.float32)
filter_data = tf.Variable(np.random.rand(3,3,4,2),dtype=np.float32)

# y = tf.nn.conv2d(input_data,filter_data,strides=[1,1,1,1],padding='VALID')
y = tf.nn.conv2d(input_data,filter_data,strides=[1,1,1,1],padding='SAME')

print(input_data)
print(y)
