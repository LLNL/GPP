# Copyright 2020 Lawrence Livermore National Security, LLC and other authors: Rushil Anirudh, Suhas Lohit, Pavan Turaga
# SPDX-License-Identifier: MIT

import numpy as np
import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

import matplotlib.gridspec as gridspec
from scipy import signal
import tensorflow as tf
import scipy
# from skimage.filters import gaussian, sobel, sobel_h,sobel_v
from skimage.io import imsave
from skimage.transform import resize


def block_diagonal(matrices, dtype=tf.float32):
  r"""Constructs block-diagonal matrices from a list of batched 2D tensors.

  Args:
    matrices: A list of Tensors with shape [..., N_i, M_i] (i.e. a list of
      matrices with the same batch dimension).
    dtype: Data type to use. The Tensors in `matrices` must match this dtype.
  Returns:
    A matrix with the input matrices stacked along its main diagonal, having
    shape [..., \sum_i N_i, \sum_i M_i].

  """
  matrices = [tf.convert_to_tensor(matrix, dtype=dtype) for matrix in matrices]
  blocked_rows = tf.Dimension(0)
  blocked_cols = tf.Dimension(0)
  batch_shape = tf.TensorShape(None)
  for matrix in matrices:
    full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
    batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
    blocked_rows += full_matrix_shape[-2]
    blocked_cols += full_matrix_shape[-1]
  ret_columns_list = []
  for matrix in matrices:
    matrix_shape = tf.shape(matrix)
    ret_columns_list.append(matrix_shape[-1])
  ret_columns = tf.add_n(ret_columns_list)
  row_blocks = []
  current_column = 0
  for matrix in matrices:
    matrix_shape = tf.shape(matrix)
    row_before_length = current_column
    current_column += matrix_shape[-1]
    row_after_length = ret_columns - current_column
    row_blocks.append(tf.pad(
        tensor=matrix,
        paddings=tf.concat(
            [tf.zeros([tf.rank(matrix) - 1, 2], dtype=tf.int32),
             [(row_before_length, row_after_length)]],
            axis=0)))
  blocked = tf.concat(row_blocks, -2)
  blocked.set_shape(batch_shape.concatenate((blocked_rows, blocked_cols)))
  return blocked

def spectral_norm(w, name,iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable(name, [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):

        """
        power iteration
        Usually iteration = 1 will be enough
        """

        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)


    return w_norm


def sample_Z(m, n):
    return np.random.uniform(-1,1,size=[m, n])


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def weight_variable_xavier_initialized(shape, constant=1, name=None):
    stddev = constant * np.sqrt(2.0 / (shape[2] + shape[3]))
    return weight_variable(shape, stddev=stddev, name=name)

def weight_variable(shape, stddev=0.02, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial,regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))

def conv2d_transpose_strided(x, W, b, output_shape=None):
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 2, 2, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)

def conv2d(x, W,strides=(1,2,2,1)):
  return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)

def leaky_relu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def bn(x,is_training,name):
     return batch_norm(x, decay=0.9, center=True, scale=True,updates_collections=None,is_training=is_training,
     reuse=None,
     trainable=True,
     scope=name)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
     c = images.shape[3]
     img = np.zeros((h * size[0], w * size[1], c))
     for idx, image in enumerate(images):
       i = idx % size[1]
       j = idx // size[1]
       img[j * h:j * h + h, i * w:i * w + w, :] = image
     return img
    elif images.shape[3]==1:
     img = np.zeros((h * size[0], w * size[1]))
     for idx, image in enumerate(images):
       i = idx % size[1]
       j = idx // size[1]
       img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
     return img
    else:
     raise ValueError('in merge(images,size) images parameter '
                      'must have dimensions: HxW or HxWx3 or HxWx4')

def grid_imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return imsave(path, image)
