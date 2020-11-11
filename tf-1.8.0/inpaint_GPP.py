# Copyright 2020 Lawrence Livermore National Security, LLC and other authors: Rushil Anirudh, Suhas Lohit, Pavan Turaga
# SPDX-License-Identifier: MIT
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from model import *
from utils import block_diagonal

import matplotlib.gridspec as gridspec
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import pybm3d
from skimage.transform import rescale, resize
from skimage import color
from skimage import io

def projector_tf(imgs,phi=None):
    csproj = tf.matmul(imgs,tf.squeeze(phi))
    return csproj

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

def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return scipy.misc.imsave(path, image)

def sample_Z(m, n):
    return np.random.uniform(-1,1,size=[m, n])

test_image = 'color_car'
id = 0
I_y = 640
I_x = 800
d_x = d_y = 32
dim_x = d_x*d_y
batch_size = (I_x*I_y)//(dim_x)
n_measure = 0.02
dim_z = 100
n_img_plot_x = I_x//d_x
n_img_plot_y = I_y//d_y

dim_z = 100
nIter = 501
iters = np.array(np.geomspace(10,10,nIter),dtype=int)

modelsave ='./all_models/gen_models_corrupt-cifar32'
fname = '/p/lustre1/anirudh1/GAN/mimicGAN/IMAGENET/test_images/{}.jpg'.format(test_image)

image = io.imread(fname,as_gray=True)
x_test = resize(image, (I_x, I_y),anti_aliasing=True,preserve_range=True,mode='reflect')
x_test_ = np.array(x_test)/np.max(x_test)
print(x_test_.shape)

mask_inpaint = np.random.rand(I_x,I_y)
mask_inpaint = np.where(mask_inpaint>(1-n_measure),1,0)
m_batch = []
x_test = []
for i in range(n_img_plot_x):
    for j in range(n_img_plot_y):
        _x = x_test_[i*d_x:d_x*(i+1),j*d_y:d_y*(j+1)]
        _m = mask_inpaint[i*d_x:d_x*(i+1),j*d_y:d_y*(j+1)]
        x_test.append(_x)
        m_batch.append(_m)


m_batch = np.expand_dims(np.array(m_batch),axis=3)
x_test = np.expand_dims(np.array(x_test),axis=3)
x_test = np.multiply(x_test,m_batch)
print(x_test.shape)
test_images = x_test[:batch_size,:,:,:]
scipy.misc.imsave('cs_outs/gt.png',x_test_)
imsave(test_images,[n_img_plot_x,n_img_plot_y],'cs_outs/inpaint.png')


tf.reset_default_graph()
tf.set_random_seed(0)
np.random.seed(4321)

obs_ph = tf.placeholder(tf.float32,[batch_size,d_x,d_y,1])
mask_ph = tf.placeholder(tf.float32,[batch_size,d_y,d_x,1])

tmp = tf.random_uniform([1000,dim_z],minval=-1.,maxval=1.)
tmp = tf.expand_dims(tf.reduce_mean(tmp,axis=0),axis=0)

z_ = tf.tile(tmp,[batch_size,1])
z_prior_ = tf.Variable(z_,name="z_prior")

G_sample = 0.5*generator(z_prior_,False)+0.5
G_sample = tf.image.resize_images(G_sample,[d_x,d_y],tf.image.ResizeMethod.BICUBIC)

est_obs = tf.multiply(mask_ph,G_sample)

loss = 1e3*tf.reduce_mean(tf.square(est_obs-obs_ph))
G_loss = 1e3*tf.reduce_mean(tf.square(est_obs-obs_ph))
z_clip = tf.clip_by_value(z_prior_,-1,1)

t_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
g_vars = [var for var in t_vars if 'Generator' in var.name]


solution_opt = tf.train.RMSPropOptimizer(5e-3).minimize(loss, var_list=[z_prior_])


saver = tf.train.Saver(g_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(modelsave)


    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("************ Prior restored! **************")


    y_obs = test_images

    for i in xrange(nIter):

        if i %10 ==0:
            G_imgs,tr_loss = sess.run([G_sample,G_loss],feed_dict={mask_ph:m_batch,obs_ph:y_obs})
            print('iter: {:d}, tr loss: {:.4f}'.format(i,tr_loss))
            merged = merge(G_imgs,[n_img_plot_x,n_img_plot_y])
            # merged_clean = pybm3d.bm3d.bm3d(merged,0.5)
            scipy.misc.imsave('cs_outs/inv_solution_{}.png'.format(str(i).zfill(3)),merged)


        fd = {mask_ph:m_batch,obs_ph:y_obs}
        for j in range(iters[i]):
            _,_,tr_loss = sess.run([z_clip,solution_opt,G_loss],feed_dict=fd)
