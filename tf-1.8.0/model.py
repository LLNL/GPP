from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
tf.set_random_seed(0)
from utils import *
from utils import spectral_norm as SN




np.random.seed(100)

'''
code adapted from https://github.com/sugyan/tf-dcgan/blob/master/dcgan.py
'''
def sample2_Z(m, n):
    return np.random.uniform(-1,1,size=[m, n])

def generator_DIP(z,reuse=None,dropout=0.05):
    print('Generator')
    # dropout = 0.05
    with tf.variable_scope('Generator',reuse=reuse):
           # reshape from inputs
           w_fc1 = weight_variable([100,120*4*4],stddev=0.02, name="w_fc1")
           b_fc1 = bias_variable([120*4*4], name="b_fc1")
           h_fc1 = tf.nn.relu(tf.matmul(z, w_fc1) + b_fc1)
           h_reshaped = tf.nn.dropout(tf.reshape(h_fc1, [-1, 4, 4, 120]),keep_prob=1-dropout)

           W_conv_t1 = weight_variable_xavier_initialized([3, 3, 25, 120], name="W_conv_t1")
           b_conv_t1 = bias_variable([25], name="b_conv_t1")
           deconv_shape = tf.stack([tf.shape(h_reshaped)[0], 8, 8, 25])
           h_conv_t1 = conv2d_transpose_strided(h_reshaped, W_conv_t1, b_conv_t1,output_shape=deconv_shape)
           h_relu_t1 = tf.nn.dropout(tf.nn.relu(h_conv_t1),keep_prob = 1-dropout)

           W_conv_t2 = weight_variable_xavier_initialized([3, 3, 15, 25], name="W_conv_t2")
           b_conv_t2 = bias_variable([15], name="b_conv_t2")
           deconv_shape = tf.stack([tf.shape(h_reshaped)[0], 16, 16, 15])
           h_conv_t2 = conv2d_transpose_strided(h_relu_t1, W_conv_t2, b_conv_t2,output_shape=deconv_shape)
           h_relu_t2 = tf.nn.dropout(tf.nn.relu(h_conv_t2),keep_prob=1-dropout)

           W_conv_t3 = weight_variable_xavier_initialized([3, 3, 1, 15], name="W_conv_t3")
           b_conv_t3 = bias_variable([1], name="b_conv_t3")
           deconv_shape = tf.stack([tf.shape(h_reshaped)[0], 32, 32, 1])
           h_conv_t3 = conv2d_transpose_strided(h_relu_t2, W_conv_t3, b_conv_t3,output_shape=deconv_shape)
           h_relu_t3 = tf.nn.tanh(h_conv_t3)

           # W_conv_t4 = weight_variable_xavier_initialized([5, 5, 1, 10], name="W_conv_t4")
           # b_conv_t4 = bias_variable([1], name="b_conv_t4")
           # deconv_shape = tf.stack([tf.shape(h_reshaped)[0], 64, 64, 1])
           # h_conv_t4 = conv2d_transpose_strided(h_relu_t3, W_conv_t4, b_conv_t4,output_shape=deconv_shape)

           outputs = h_relu_t3

    return outputs

def generator_c(z,train_mode,dim_z=100,reuse=None):
    print('Generator')
    with tf.variable_scope('Generator-color',reuse=reuse):
           # reshape from inputs
           w_fc1 = weight_variable([dim_z,512*4*4],stddev=0.02, name="w_fc1")
           b_fc1 = bias_variable([512*4*4], name="b_fc1")
           h_fc1 = bn(tf.nn.relu(tf.matmul(z, w_fc1) + b_fc1), train_mode,"bn1")
           h_reshaped = tf.reshape(h_fc1, [-1, 4, 4, 512])

           W_conv_t1 = weight_variable_xavier_initialized([5, 5, 256, 512], name="W_conv_t1")
           b_conv_t1 = bias_variable([256], name="b_conv_t1")
           deconv_shape = tf.stack([tf.shape(h_reshaped)[0], 8, 8, 256])
           h_conv_t1 = conv2d_transpose_strided(h_reshaped, W_conv_t1, b_conv_t1,output_shape=deconv_shape)
           h_relu_t1 = bn(tf.nn.relu(h_conv_t1),train_mode,"bn2")

           W_conv_t2 = weight_variable_xavier_initialized([5, 5, 128, 256], name="W_conv_t2")
           b_conv_t2 = bias_variable([128], name="b_conv_t2")
           deconv_shape = tf.stack([tf.shape(h_reshaped)[0], 16, 16, 128])
           h_conv_t2 = conv2d_transpose_strided(h_relu_t1, W_conv_t2, b_conv_t2,output_shape=deconv_shape)
           h_relu_t2 = bn(tf.nn.relu(h_conv_t2),train_mode,"bn3")

           W_conv_t3 = weight_variable_xavier_initialized([5, 5, 3, 128], name="W_conv_t3")
           b_conv_t3 = bias_variable([3], name="b_conv_t3")
           deconv_shape = tf.stack([tf.shape(h_reshaped)[0], 32, 32, 3])
           h_conv_t3 = conv2d_transpose_strided(h_relu_t2, W_conv_t3, b_conv_t3,output_shape=deconv_shape)

           outputs = tf.nn.tanh(h_conv_t3)

    return outputs


def generator(z,train_mode,reuse=None):
    print('Generator')
    with tf.variable_scope('Generator',reuse=reuse):
           # reshape from inputs
           w_fc1 = weight_variable([100,512*4*4],stddev=0.02, name="w_fc1")
           b_fc1 = bias_variable([512*4*4], name="b_fc1")
           h_fc1 = bn(tf.nn.relu(tf.matmul(z, w_fc1) + b_fc1), train_mode,"bn1")
           h_reshaped = tf.reshape(h_fc1, [-1, 4, 4, 512])

           W_conv_t1 = weight_variable_xavier_initialized([5, 5, 256, 512], name="W_conv_t1")
           b_conv_t1 = bias_variable([256], name="b_conv_t1")
           deconv_shape = tf.stack([tf.shape(h_reshaped)[0], 8, 8, 256])
           h_conv_t1 = conv2d_transpose_strided(h_reshaped, W_conv_t1, b_conv_t1,output_shape=deconv_shape)
           h_relu_t1 = bn(tf.nn.relu(h_conv_t1),train_mode,"bn2")

           W_conv_t2 = weight_variable_xavier_initialized([5, 5, 128, 256], name="W_conv_t2")
           b_conv_t2 = bias_variable([128], name="b_conv_t2")
           deconv_shape = tf.stack([tf.shape(h_reshaped)[0], 16, 16, 128])
           h_conv_t2 = conv2d_transpose_strided(h_relu_t1, W_conv_t2, b_conv_t2,output_shape=deconv_shape)
           h_relu_t2 = bn(tf.nn.relu(h_conv_t2),train_mode,"bn3")

           W_conv_t3 = weight_variable_xavier_initialized([5, 5, 1, 128], name="W_conv_t3")
           b_conv_t3 = bias_variable([1], name="b_conv_t3")
           deconv_shape = tf.stack([tf.shape(h_reshaped)[0], 32, 32, 1])
           h_conv_t3 = conv2d_transpose_strided(h_relu_t2, W_conv_t3, b_conv_t3,output_shape=deconv_shape)
           h_relu_t3 = bn(tf.nn.relu(h_conv_t3),train_mode,"bn4")

           # W_conv_t4 = weight_variable_xavier_initialized([5, 5, 3, 64], name="W_conv_t4")
           # b_conv_t4 = bias_variable([3], name="b_conv_t4")
           # deconv_shape = tf.stack([tf.shape(h_reshaped)[0], 32, 32, 3])
           # h_conv_t4 = conv2d_transpose_strided(h_relu_t3, W_conv_t4, b_conv_t4,output_shape=deconv_shape)
           outputs = tf.nn.tanh(h_conv_t3)

    return outputs


def discriminator(data,train_mode,reuse=None):
    # outputs = tf.convert_to_tensor(data)

    print('Discriminator')
    with tf.variable_scope('Discriminator', reuse=reuse):
        W_conv1 = weight_variable_xavier_initialized([5,5,1,64],name="d_w_conv1")
        b_conv1 = bias_variable([64],name="d_b_conv1")
        h_conv1 = conv2d(data, W_conv1) + b_conv1
        h_relu1 = lrelu(bn(h_conv1,train_mode,name="d_bn1"))

        W_conv2 = weight_variable_xavier_initialized([5,5,64,128],name="d_w_conv2")
        b_conv2 = bias_variable([128],name="d_b_conv2")
        h_conv2 = conv2d(h_relu1, W_conv2) + b_conv2
        h_relu2 = lrelu(bn(h_conv2,train_mode,name="d_bn2"))

        W_conv3 = weight_variable_xavier_initialized([5,5,128,256],name="d_w_conv3")
        b_conv3 = bias_variable([256],name="d_b_conv3")
        h_conv3 = conv2d(h_relu2, W_conv3) + b_conv3
        h_relu3 = lrelu(bn(h_conv3,train_mode,name="d_bn3"))

        W_conv4 = weight_variable_xavier_initialized([5,5,256,512],name="d_w_conv4")
        b_conv4 = bias_variable([512],name="d_b_conv4")
        h_conv4 = conv2d(h_relu3, W_conv4) + b_conv4
        h_relu4 = lrelu(bn(h_conv4,train_mode,name="d_bn4"))

        batch_size = tf.shape(h_relu4)[0]
        reshape = tf.reshape(h_relu4, [batch_size, 2*2*512])

        w_fc1 = weight_variable([2*2*512, 1], name="d_w_fc1")
        b_fc1 = bias_variable([1], name="d_b_fc1")
        outputs = tf.matmul(reshape, w_fc1) + b_fc1
        d_prob = tf.nn.sigmoid(outputs)

    return outputs, d_prob

def discriminator_c(data,train_mode,reuse=None):
    # outputs = tf.convert_to_tensor(data)

    print('Discriminator')
    with tf.variable_scope('Discriminator', reuse=reuse):
        W_conv1 = weight_variable_xavier_initialized([5,5,3,64],name="d_w_conv1")
        b_conv1 = bias_variable([64],name="d_b_conv1")
        h_conv1 = conv2d(data, W_conv1) + b_conv1
        h_relu1 = lrelu(bn(h_conv1,train_mode,name="d_bn1"))

        W_conv2 = weight_variable_xavier_initialized([5,5,64,128],name="d_w_conv2")
        b_conv2 = bias_variable([128],name="d_b_conv2")
        h_conv2 = conv2d(h_relu1, W_conv2) + b_conv2
        h_relu2 = lrelu(bn(h_conv2,train_mode,name="d_bn2"))

        W_conv3 = weight_variable_xavier_initialized([5,5,128,256],name="d_w_conv3")
        b_conv3 = bias_variable([256],name="d_b_conv3")
        h_conv3 = conv2d(h_relu2, W_conv3) + b_conv3
        h_relu3 = lrelu(bn(h_conv3,train_mode,name="d_bn3"))

        W_conv4 = weight_variable_xavier_initialized([5,5,256,512],name="d_w_conv4")
        b_conv4 = bias_variable([512],name="d_b_conv4")
        h_conv4 = conv2d(h_relu3, W_conv4) + b_conv4
        h_relu4 = lrelu(bn(h_conv4,train_mode,name="d_bn4"))

        batch_size = tf.shape(h_relu4)[0]
        reshape = tf.reshape(h_relu4, [batch_size, 2*2*512])

        w_fc1 = weight_variable([2*2*512, 1], name="d_w_fc1")
        b_fc1 = bias_variable([1], name="d_b_fc1")
        outputs = tf.matmul(reshape, w_fc1) + b_fc1
        d_prob = tf.nn.sigmoid(outputs)

    return outputs, d_prob
